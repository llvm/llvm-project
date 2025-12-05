/**
 * @file dsmil-telemetry-summary.cpp
 * @brief DSLLVM Telemetry Summary Tool
 *
 * Aggregates telemetry metrics from all modules and generates a global summary.
 * Reads *.dsmil.metrics.json files and existing .telemetry.json files.
 *
 * Usage:
 *   dsmil-telemetry-summary [options]
 *   Options:
 *     --input-glob <pattern>  : Glob pattern for input JSON files (default: "*.dsmil.metrics.json")
 *     --output <path>         : Output path for global metrics (default: "dsmil.global.metrics.json")
 *     --telemetry-json <glob> : Glob pattern for telemetry JSON files (optional)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <glob.h>
#include <cstring>
#include <cstdlib>
#include <cmath>

namespace fs = std::filesystem;

struct GlobalMetrics {
    // Overall statistics
    size_t total_modules = 0;
    size_t total_functions = 0;
    size_t total_instrumented = 0;
    double avg_coverage = 0.0;
    
    // Category totals
    std::map<std::string, size_t> category_totals;
    
    // OT-specific totals
    size_t total_ot_critical = 0;
    size_t total_ses_gates = 0;
    
    // Generic annotation totals
    size_t total_net_io = 0;
    size_t total_crypto = 0;
    size_t total_process = 0;
    size_t total_file = 0;
    size_t total_untrusted = 0;
    size_t total_error_handler = 0;
    
    // Authority tier totals
    size_t total_tier_0 = 0;
    size_t total_tier_1 = 0;
    size_t total_tier_2 = 0;
    size_t total_tier_3 = 0;
    
    // Telecom totals
    size_t total_telecom = 0;
    std::map<std::string, size_t> telecom_stacks;
    std::map<std::string, size_t> ss7_roles;
    std::map<std::string, size_t> sigtran_roles;
    std::map<std::string, size_t> telecom_envs;
    
    // Safety signals
    size_t total_safety_signals = 0;
    
    // Layer/device distribution
    std::map<uint8_t, size_t> layer_totals;
    std::map<uint8_t, size_t> device_totals;
    
    // Mission profile distribution
    std::map<std::string, size_t> mission_profiles;
    
    // Telemetry level distribution (if available)
    std::map<std::string, size_t> telemetry_levels;
};

/**
 * Simple JSON parser (minimal implementation)
 */
class SimpleJSONParser {
private:
    std::string json;
    size_t pos = 0;
    
    void skipWhitespace() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || 
                                     json[pos] == '\r' || json[pos] == '\t')) {
            pos++;
        }
    }
    
    std::string parseString() {
        skipWhitespace();
        if (pos >= json.size() || json[pos] != '"') return "";
        pos++;  // skip opening quote
        
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == 'n') result += '\n';
                else if (json[pos] == 't') result += '\t';
                else if (json[pos] == '\\') result += '\\';
                else if (json[pos] == '"') result += '"';
                else result += json[pos];
            } else {
                result += json[pos];
            }
            pos++;
        }
        if (pos < json.size()) pos++;  // skip closing quote
        return result;
    }
    
    double parseNumber() {
        skipWhitespace();
        size_t start = pos;
        bool hasDot = false;
        while (pos < json.size() && 
               ((json[pos] >= '0' && json[pos] <= '9') || json[pos] == '.' || json[pos] == '-')) {
            if (json[pos] == '.') hasDot = true;
            pos++;
        }
        std::string numStr = json.substr(start, pos - start);
        return std::stod(numStr);
    }
    
    size_t parseInteger() {
        return (size_t)parseNumber();
    }
    
public:
    SimpleJSONParser(const std::string &jsonStr) : json(jsonStr) {}
    
    std::map<std::string, double> parseObject() {
        std::map<std::string, double> result;
        skipWhitespace();
        if (pos >= json.size() || json[pos] != '{') return result;
        pos++;  // skip opening brace
        
        while (pos < json.size()) {
            skipWhitespace();
            if (json[pos] == '}') {
                pos++;
                break;
            }
            
            std::string key = parseString();
            skipWhitespace();
            if (pos >= json.size() || json[pos] != ':') break;
            pos++;  // skip colon
            
            skipWhitespace();
            if (json[pos] == '"') {
                // String value - skip for now
                parseString();
            } else if (json[pos] == '{') {
                // Nested object - skip
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '{') depth++;
                    else if (json[pos] == '}') depth--;
                    pos++;
                }
            } else if (json[pos] == '[') {
                // Array - skip
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '[') depth++;
                    else if (json[pos] == ']') depth--;
                    pos++;
                }
            } else {
                double value = parseNumber();
                result[key] = value;
            }
            
            skipWhitespace();
            if (pos < json.size() && json[pos] == ',') {
                pos++;
            } else if (pos < json.size() && json[pos] == '}') {
                pos++;
                break;
            }
        }
        return result;
    }
    
    double getValue(const std::string &path) {
        // Simple path lookup like "metrics.total_functions"
        size_t oldPos = pos;
        pos = 0;
        
        std::vector<std::string> parts;
        size_t start = 0;
        for (size_t i = 0; i <= path.size(); i++) {
            if (i == path.size() || path[i] == '.') {
                if (i > start) {
                    parts.push_back(path.substr(start, i - start));
                }
                start = i + 1;
            }
        }
        
        // Navigate through JSON
        for (size_t i = 0; i < parts.size(); i++) {
            skipWhitespace();
            if (pos >= json.size() || json[pos] != '{') {
                pos = oldPos;
                return 0.0;
            }
            pos++;
            
            // Find key
            while (pos < json.size()) {
                skipWhitespace();
                if (json[pos] == '}') {
                    pos = oldPos;
                    return 0.0;
                }
                
                std::string key = parseString();
                skipWhitespace();
                if (pos >= json.size() || json[pos] != ':') break;
                pos++;
                
                if (key == parts[i]) {
                    if (i == parts.size() - 1) {
                        // Last part - return value
                        double value = parseNumber();
                        pos = oldPos;
                        return value;
                    } else {
                        // Continue to nested object
                        skipWhitespace();
                        if (json[pos] != '{') {
                            pos = oldPos;
                            return 0.0;
                        }
                        break;
                    }
                } else {
                    // Skip this value
                    skipWhitespace();
                    if (json[pos] == '"') parseString();
                    else if (json[pos] == '{') {
                        int depth = 1;
                        pos++;
                        while (pos < json.size() && depth > 0) {
                            if (json[pos] == '{') depth++;
                            else if (json[pos] == '}') depth--;
                            pos++;
                        }
                    } else if (json[pos] == '[') {
                        int depth = 1;
                        pos++;
                        while (pos < json.size() && depth > 0) {
                            if (json[pos] == '[') depth++;
                            else if (json[pos] == ']') depth--;
                            pos++;
                        }
                    } else {
                        parseNumber();
                    }
                    
                    skipWhitespace();
                    if (pos < json.size() && json[pos] == ',') pos++;
                }
            }
        }
        
        pos = oldPos;
        return 0.0;
    }
};

/**
 * Parse metrics JSON file
 */
bool parseMetricsJSON(const std::string &filepath, GlobalMetrics &global) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file: " << filepath << std::endl;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    SimpleJSONParser parser(content);
    
    // Extract values
    global.total_modules++;
    global.total_functions += (size_t)parser.getValue("metrics.total_functions");
    global.total_instrumented += (size_t)parser.getValue("metrics.instrumented_functions");
    global.total_ot_critical += (size_t)parser.getValue("metrics.ot_critical_count");
    global.total_ses_gates += (size_t)parser.getValue("metrics.ses_gate_count");
    global.total_net_io += (size_t)parser.getValue("metrics.net_io_count");
    global.total_crypto += (size_t)parser.getValue("metrics.crypto_count");
    global.total_process += (size_t)parser.getValue("metrics.process_count");
    global.total_file += (size_t)parser.getValue("metrics.file_count");
    global.total_untrusted += (size_t)parser.getValue("metrics.untrusted_count");
    global.total_error_handler += (size_t)parser.getValue("metrics.error_handler_count");
    global.total_tier_0 += (size_t)parser.getValue("metrics.authority_tiers.tier_0");
    global.total_tier_1 += (size_t)parser.getValue("metrics.authority_tiers.tier_1");
    global.total_tier_2 += (size_t)parser.getValue("metrics.authority_tiers.tier_2");
    global.total_tier_3 += (size_t)parser.getValue("metrics.authority_tiers.tier_3");
    global.total_telecom += (size_t)parser.getValue("metrics.telecom.total");
    global.total_safety_signals += (size_t)parser.getValue("metrics.safety_signals");
    
    // Extract mission profile
    std::string mission_profile;
    size_t profileStart = content.find("\"mission_profile\": \"");
    if (profileStart != std::string::npos) {
        profileStart += 20;
        size_t profileEnd = content.find("\"", profileStart);
        if (profileEnd != std::string::npos) {
            mission_profile = content.substr(profileStart, profileEnd - profileStart);
            global.mission_profiles[mission_profile]++;
        }
    }
    
    return true;
}

/**
 * Expand glob pattern to file list
 */
std::vector<std::string> expandGlob(const std::string &pattern) {
    std::vector<std::string> files;
    
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    
    int ret = glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &glob_result);
    if (ret == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; i++) {
            files.push_back(glob_result.gl_pathv[i]);
        }
    }
    
    globfree(&glob_result);
    return files;
}

/**
 * Generate global metrics JSON
 */
void generateGlobalMetricsJSON(const GlobalMetrics &metrics, const std::string &outputPath) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open output file: " << outputPath << std::endl;
        return;
    }
    
    out << "{\n";
    out << "  \"summary\": {\n";
    out << "    \"total_modules\": " << metrics.total_modules << ",\n";
    out << "    \"total_functions\": " << metrics.total_functions << ",\n";
    out << "    \"total_instrumented\": " << metrics.total_instrumented << ",\n";
    out << "    \"avg_coverage\": " 
        << (metrics.total_functions > 0 ? 
            (100.0 * metrics.total_instrumented / metrics.total_functions) : 0.0)
        << ",\n";
    out << "    \"total_ot_critical\": " << metrics.total_ot_critical << ",\n";
    out << "    \"total_ses_gates\": " << metrics.total_ses_gates << ",\n";
    out << "    \"total_net_io\": " << metrics.total_net_io << ",\n";
    out << "    \"total_crypto\": " << metrics.total_crypto << ",\n";
    out << "    \"total_process\": " << metrics.total_process << ",\n";
    out << "    \"total_file\": " << metrics.total_file << ",\n";
    out << "    \"total_untrusted\": " << metrics.total_untrusted << ",\n";
    out << "    \"total_error_handler\": " << metrics.total_error_handler << ",\n";
    out << "    \"total_safety_signals\": " << metrics.total_safety_signals << "\n";
    out << "  },\n";
    
    out << "  \"authority_tiers\": {\n";
    out << "    \"tier_0\": " << metrics.total_tier_0 << ",\n";
    out << "    \"tier_1\": " << metrics.total_tier_1 << ",\n";
    out << "    \"tier_2\": " << metrics.total_tier_2 << ",\n";
    out << "    \"tier_3\": " << metrics.total_tier_3 << "\n";
    out << "  },\n";
    
    out << "  \"telecom\": {\n";
    out << "    \"total\": " << metrics.total_telecom << ",\n";
    out << "    \"stacks\": {\n";
    bool first = true;
    for (const auto &pair : metrics.telecom_stacks) {
        if (!first) out << ",\n";
        out << "      \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n    },\n";
    out << "    \"ss7_roles\": {\n";
    first = true;
    for (const auto &pair : metrics.ss7_roles) {
        if (!first) out << ",\n";
        out << "      \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n    },\n";
    out << "    \"sigtran_roles\": {\n";
    first = true;
    for (const auto &pair : metrics.sigtran_roles) {
        if (!first) out << ",\n";
        out << "      \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n    },\n";
    out << "    \"environments\": {\n";
    first = true;
    for (const auto &pair : metrics.telecom_envs) {
        if (!first) out << ",\n";
        out << "      \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n    }\n";
    out << "  },\n";
    
    out << "  \"mission_profiles\": {\n";
    first = true;
    for (const auto &pair : metrics.mission_profiles) {
        if (!first) out << ",\n";
        out << "      \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n  },\n";
    
    out << "  \"layers\": {\n";
    first = true;
    for (const auto &pair : metrics.layer_totals) {
        if (!first) out << ",\n";
        out << "      \"" << (int)pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n  },\n";
    
    out << "  \"devices\": {\n";
    first = true;
    for (const auto &pair : metrics.device_totals) {
        if (!first) out << ",\n";
        out << "      \"" << (int)pair.first << "\": " << pair.second;
        first = false;
    }
    out << "\n  }\n";
    
    out << "}\n";
    out.close();
    
    std::cout << "Generated global metrics: " << outputPath << std::endl;
}

int main(int argc, char *argv[]) {
    std::string inputGlob = "*.dsmil.metrics.json";
    std::string outputPath = "dsmil.global.metrics.json";
    std::string telemetryJsonGlob = "";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input-glob") == 0 && i + 1 < argc) {
            inputGlob = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (strcmp(argv[i], "--telemetry-json") == 0 && i + 1 < argc) {
            telemetryJsonGlob = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: dsmil-telemetry-summary [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --input-glob <pattern>  : Glob pattern for metrics JSON files\n";
            std::cout << "                          (default: *.dsmil.metrics.json)\n";
            std::cout << "  --output <path>         : Output path for global metrics\n";
            std::cout << "                          (default: dsmil.global.metrics.json)\n";
            std::cout << "  --telemetry-json <glob> : Glob pattern for telemetry JSON files\n";
            std::cout << "  --help, -h              : Show this help message\n";
            return 0;
        }
    }
    
    GlobalMetrics globalMetrics;
    
    // Process metrics JSON files
    std::cout << "Processing metrics files matching: " << inputGlob << std::endl;
    std::vector<std::string> metricsFiles = expandGlob(inputGlob);
    
    for (const auto &file : metricsFiles) {
        std::cout << "  Reading: " << file << std::endl;
        parseMetricsJSON(file, globalMetrics);
    }
    
    // Process telemetry JSON files if specified
    if (!telemetryJsonGlob.empty()) {
        std::cout << "Processing telemetry files matching: " << telemetryJsonGlob << std::endl;
        std::vector<std::string> telemetryFiles = expandGlob(telemetryJsonGlob);
        // For now, just count files (could parse event counts, etc.)
        std::cout << "  Found " << telemetryFiles.size() << " telemetry files\n";
    }
    
    // Generate summary
    std::cout << "\nSummary:\n";
    std::cout << "  Modules: " << globalMetrics.total_modules << "\n";
    std::cout << "  Total Functions: " << globalMetrics.total_functions << "\n";
    std::cout << "  Instrumented: " << globalMetrics.total_instrumented << "\n";
    std::cout << "  Coverage: " 
              << (globalMetrics.total_functions > 0 ? 
                  (100.0 * globalMetrics.total_instrumented / globalMetrics.total_functions) : 0.0)
              << "%\n";
    
    generateGlobalMetricsJSON(globalMetrics, outputPath);
    
    return 0;
}
