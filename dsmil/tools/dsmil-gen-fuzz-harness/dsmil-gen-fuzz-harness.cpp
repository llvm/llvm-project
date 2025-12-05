/**
 * @file dsmil-gen-fuzz-harness.cpp
 * @brief DSLLVM General-Purpose Fuzz Harness Generator
 *
 * Generates libFuzzer/AFL++ harnesses for any fuzzing target.
 * Supports grammar-based, ML-guided, structure-aware, and distributed fuzzing.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <sstream>

struct FuzzTargetConfig {
    std::string type;                      // "generic", "protocol", "parser", "api", etc.
    std::string name;                      // Target name
    
    // Input handling
    std::string input_format;              // "binary", "text", "structured"
    size_t max_input_size = 1048576;      // 1MB default
    
    // Advanced options
    bool enable_grammar_fuzzing = false;
    bool enable_structure_aware = false;
    bool enable_ml_guided = false;
    bool enable_dictionary = false;
    bool enable_distributed = false;
    bool enable_perf_counters = false;
    
    // Grammar options
    std::string grammar_file;
    std::vector<std::string> grammar_rules;
    
    // Dictionary options
    std::vector<std::string> dictionary_entries;
    std::string dictionary_file;
    
    // ML options
    std::string ml_model_path;
    bool ml_online = false;
    std::string ml_endpoint;
    
    // Distributed options
    uint32_t worker_id = 0;
    uint32_t num_workers = 1;
    std::string corpus_sync_path;
    
    // Performance options
    bool enable_parallel = false;
    uint32_t num_threads = 1;
    bool enable_batch_processing = false;
    size_t batch_size = 1000;
    
    // Target-specific options (key-value pairs)
    std::map<std::string, std::string> target_options;
};

void generate_generic_harness(const FuzzTargetConfig &config, std::ostream &out) {
    out << "// Generated General-Purpose Fuzz Harness\n";
    out << "// Target: " << config.name << " (" << config.type << ")\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include <string.h>\n";
    out << "#include <vector>\n";
    out << "#include <unordered_set>\n";
    out << "#include \"dsmil_fuzz_telemetry.h\"\n";
    out << "#include \"dsmil_fuzz_telemetry_advanced.h\"\n";
    out << "\n";
    
    if (config.enable_grammar_fuzzing) {
        out << "// Grammar-based generation\n";
        out << "struct GrammarGenerator {\n";
        out << "    std::vector<uint8_t> generate(const uint8_t *seed, size_t seed_len);\n";
        out << "};\n\n";
    }
    
    if (config.enable_dictionary) {
        out << "// Dictionary for structure-aware fuzzing\n";
        out << "static const char* fuzz_dictionary[] = {\n";
        for (size_t i = 0; i < config.dictionary_entries.size() && i < 50; i++) {
            out << "    \"" << config.dictionary_entries[i] << "\",\n";
        }
        out << "    nullptr\n";
        out << "};\n\n";
    }
    
    out << "// Coverage tracking\n";
    out << "static std::unordered_set<uint32_t> covered_edges;\n";
    out << "static std::unordered_set<uint32_t> covered_states;\n";
    out << "static uint64_t total_executions = 0;\n";
    out << "static uint64_t unique_coverage_inputs = 0;\n\n";
    
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    // Initialize advanced telemetry\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsmil_fuzz_telemetry_advanced_init(NULL, 1048576, " 
        << (config.enable_perf_counters ? "1" : "0") << ", "
        << (config.enable_ml_guided ? "1" : "0") << ");\n";
    out << "        initialized = true;\n";
    out << "    }\n\n";
    
    out << "    total_executions++;\n\n";
    
    out << "    // Validate input size\n";
    out << "    if (size == 0 || size > " << config.max_input_size << ") return 0;\n\n";
    
    out << "    // Compute input hash\n";
    out << "    uint64_t input_hash = 0;\n";
    out << "    for (size_t i = 0; i < size && i < 16; i++) {\n";
    out << "        input_hash = (input_hash << 8) | data[i];\n";
    out << "    }\n";
    out << "    dsmil_fuzz_set_context(input_hash);\n\n";
    
    if (config.enable_grammar_fuzzing) {
        out << "    // Grammar-based generation\n";
        out << "    GrammarGenerator grammar;\n";
        out << "    std::vector<uint8_t> generated = grammar.generate(data, size);\n";
        out << "    if (!generated.empty()) {\n";
        out << "        data = generated.data();\n";
        out << "        size = generated.size();\n";
        out << "    }\n\n";
    }
    
    if (config.enable_structure_aware) {
        out << "    // Structure-aware parsing\n";
        out << "    if (size < 4) return 0;\n";
        out << "    // Parse structure based on input_format: " << config.input_format << "\n";
        if (config.input_format == "structured") {
            out << "    uint32_t magic = *(uint32_t*)data;\n";
            out << "    uint32_t length = *(uint32_t*)(data + 4);\n";
            out << "    if (length > size - 8) return 0;\n";
        }
        out << "\n";
    }
    
    if (config.enable_ml_guided) {
        out << "    // ML-guided mutation suggestions\n";
        out << "    dsmil_mutation_metadata_t suggestions[10];\n";
        out << "    size_t num_suggestions = dsmil_fuzz_get_mutation_suggestions(\n";
        out << "        input_hash & 0xFFFFFFFF, suggestions, 10);\n";
        out << "    if (num_suggestions > 0) {\n";
        out << "        // Apply ML-suggested mutations\n";
        out << "    }\n\n";
    }
    
    out << "    // Record mutation metadata\n";
    out << "    dsmil_mutation_metadata_t mutation = {0};\n";
    out << "    mutation.strategy = DSMIL_FUZZ_STRATEGY_MUTATION;\n";
    out << "    mutation.mutation_count = 1;\n";
    out << "    mutation.mutation_type = \"input\";\n";
    out << "    dsmil_fuzz_record_mutation(&mutation);\n\n";
    
    out << "    // Process input (target-specific)\n";
    out << "    // TODO: Call your target function here\n";
    out << "    // Example: target_function(data, size);\n\n";
    
    out << "    // Track coverage\n";
    out << "    std::vector<uint32_t> new_edges;\n";
    out << "    std::vector<uint32_t> new_states;\n\n";
    
    out << "    // Update coverage map\n";
    out << "    bool new_coverage = dsmil_fuzz_update_coverage_map(\n";
    out << "        input_hash,\n";
    out << "        new_edges.data(), new_edges.size(),\n";
    out << "        new_states.data(), new_states.size());\n\n";
    
    out << "    if (new_coverage) {\n";
    out << "        unique_coverage_inputs++;\n";
    out << "        // Compute interestingness score\n";
    out << "        dsmil_coverage_feedback_t feedback = {0};\n";
    out << "        feedback.input_hash = input_hash;\n";
    out << "        feedback.new_edges = new_edges.size();\n";
    out << "        feedback.new_states = new_states.size();\n";
    out << "        double score = dsmil_fuzz_compute_interestingness(input_hash, &feedback);\n";
    out << "        if (score > 0.7) {\n";
    out << "            // High-interestingness input - save to corpus\n";
    out << "        }\n";
    out << "    }\n\n";
    
    if (config.enable_perf_counters) {
        out << "    // Record performance counters (if available)\n";
        out << "    // dsmil_fuzz_record_perf_counters(cycles, cache_misses, mispredicts);\n\n";
    }
    
    if (config.enable_distributed) {
        out << "    // Distributed fuzzing sync\n";
        out << "    if (total_executions % 1000 == 0) {\n";
        out << "        // Sync corpus with other workers\n";
        out << "    }\n\n";
    }
    
    out << "    // Periodic telemetry flush\n";
    out << "    if (total_executions % 10000 == 0) {\n";
    out << "        dsmil_fuzz_flush_advanced_events(\"telemetry.bin\", 1);\n";
    out << "    }\n\n";
    
    out << "    return 0;\n";
    out << "}\n";
}

void generate_protocol_harness(const FuzzTargetConfig &config, std::ostream &out) {
    out << "// Generated Protocol Fuzz Harness\n";
    out << "// Protocol: " << config.name << "\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include \"dsmil_fuzz_telemetry.h\"\n";
    out << "#include \"dsmil_fuzz_telemetry_advanced.h\"\n";
    out << "\n";
    
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsmil_fuzz_telemetry_advanced_init(NULL, 1048576, 0, 0);\n";
    out << "        initialized = true;\n";
    out << "    }\n\n";
    
    out << "    if (size < 4) return 0;\n\n";
    
    out << "    // Parse protocol structure\n";
    out << "    // TODO: Implement protocol-specific parsing\n\n";
    
    out << "    // Process protocol message\n";
    out << "    // TODO: Call protocol handler\n\n";
    
    out << "    return 0;\n";
    out << "}\n";
}

void generate_parser_harness(const FuzzTargetConfig &config, std::ostream &out) {
    out << "// Generated Parser Fuzz Harness\n";
    out << "// Parser: " << config.name << "\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include \"dsmil_fuzz_telemetry.h\"\n";
    out << "\n";
    
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsmil_fuzz_telemetry_init(NULL, 65536);\n";
    out << "        initialized = true;\n";
    out << "    }\n\n";
    
    out << "    if (size == 0) return 0;\n\n";
    
    out << "    // Parse input\n";
    out << "    // TODO: Call parser function\n";
    out << "    // Example: parser_parse(data, size);\n\n";
    
    out << "    return 0;\n";
    out << "}\n";
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <output.cpp> [--advanced]\n";
        return 1;
    }

    bool advanced_mode = false;
    if (argc > 3 && std::string(argv[3]) == "--advanced") {
        advanced_mode = true;
    }

    std::string config_path = argv[1];
    std::string output_path = argv[2];

    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        FuzzTargetConfig target_config;
        
        if (config["target"]) {
            auto target = config["target"];
            target_config.name = target["name"].as<std::string>("fuzz_target");
            target_config.type = target["type"].as<std::string>("generic");
            target_config.input_format = target["input_format"].as<std::string>("binary");
            
            if (target["max_input_size"]) {
                target_config.max_input_size = target["max_input_size"].as<size_t>();
            }
            
            // Advanced options
            if (target["enable_grammar_fuzzing"]) {
                target_config.enable_grammar_fuzzing = target["enable_grammar_fuzzing"].as<bool>();
            }
            if (target["grammar_file"]) {
                target_config.grammar_file = target["grammar_file"].as<std::string>();
            }
            
            if (target["enable_structure_aware"]) {
                target_config.enable_structure_aware = target["enable_structure_aware"].as<bool>();
            }
            
            if (target["enable_ml_guided"]) {
                target_config.enable_ml_guided = target["enable_ml_guided"].as<bool>();
            }
            if (target["ml_model_path"]) {
                target_config.ml_model_path = target["ml_model_path"].as<std::string>();
            }
            
            if (target["enable_dictionary"]) {
                target_config.enable_dictionary = target["enable_dictionary"].as<bool>();
            }
            if (target["dictionary"]) {
                for (auto entry : target["dictionary"]) {
                    target_config.dictionary_entries.push_back(entry.as<std::string>());
                }
            }
            
            if (target["enable_distributed"]) {
                target_config.enable_distributed = target["enable_distributed"].as<bool>();
            }
            if (target["worker_id"]) {
                target_config.worker_id = target["worker_id"].as<uint32_t>();
            }
            
            if (target["enable_perf_counters"]) {
                target_config.enable_perf_counters = target["enable_perf_counters"].as<bool>();
            }
            
            // Target-specific options
            if (target["options"]) {
                for (auto it = target["options"].begin(); it != target["options"].end(); ++it) {
                    target_config.target_options[it->first.as<std::string>()] = 
                        it->second.as<std::string>();
                }
            }
        }

        std::ofstream out(output_path);
        if (!out.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_path << "\n";
            return 1;
        }

        if (target_config.type == "protocol") {
            generate_protocol_harness(target_config, out);
        } else if (target_config.type == "parser") {
            generate_parser_harness(target_config, out);
        } else {
            generate_generic_harness(target_config, out);
        }

        std::cout << "Generated fuzz harness: " << output_path << "\n";
        std::cout << "  Target: " << target_config.name << " (" << target_config.type << ")\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
