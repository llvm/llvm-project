/**
 * @file DsmilMetricsPass.cpp
 * @brief DSLLVM Telemetry Metrics Collection Pass
 *
 * Gathers telemetry instrumentation metrics and generates JSON manifest files
 * with statistics about instrumented functions, categories, and coverage.
 *
 * Features:
 * - Total function counts
 * - Instrumented function counts by category
 * - OT tier distribution
 * - Telecom stack statistics
 * - Generic annotation statistics
 * - Coverage metrics
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#define DEBUG_TYPE "dsmil-metrics"

using namespace llvm;

// Command-line options
static cl::opt<std::string> MetricsOutputDir(
    "dsmil-metrics-output-dir",
    cl::desc("Output directory for metrics JSON files"),
    cl::init(""));

static cl::opt<std::string> MissionProfile(
    "dsmil-mission-profile",
    cl::desc("Mission profile name"),
    cl::init(""));

namespace {

/**
 * Metrics structure
 */
struct TelemetryMetrics {
    // Overall counts
    size_t total_functions = 0;
    size_t instrumented_functions = 0;
    size_t ot_critical_count = 0;
    size_t ses_gate_count = 0;
    
    // Generic annotation counts
    size_t net_io_count = 0;
    size_t crypto_count = 0;
    size_t process_count = 0;
    size_t file_count = 0;
    size_t untrusted_count = 0;
    size_t error_handler_count = 0;
    
    // OT tier distribution
    size_t tier_0_count = 0;  // Safety kernel
    size_t tier_1_count = 0;  // High-impact control
    size_t tier_2_count = 0;  // Optimization
    size_t tier_3_count = 0;  // Analytics
    
    // Telecom statistics
    size_t telecom_stack_count = 0;
    std::map<std::string, size_t> telecom_stacks;  // ss7, sigtran, sip, diameter
    std::map<std::string, size_t> ss7_roles;       // STP, MSC, HLR, etc.
    std::map<std::string, size_t> sigtran_roles;  // SG, AS, ASP, IPSP
    std::map<std::string, size_t> telecom_envs;   // prod, lab, honeypot, fuzz, sim
    
    // Safety signals
    size_t safety_signal_count = 0;
    
    // Layer/device distribution
    std::map<uint8_t, size_t> layer_counts;
    std::map<uint8_t, size_t> device_counts;
    
    // Category distribution
    std::map<std::string, size_t> category_counts;
};

/**
 * Check if function has annotation attribute (via metadata)
 */
bool hasAnnotation(Function &F, StringRef AttrName) {
    // Check for annotate metadata
    if (MDNode *MD = F.getMetadata("llvm.ptr.annotation")) {
        for (unsigned i = 0; i < MD->getNumOperands(); i++) {
            if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                if (Str->getString().startswith(AttrName)) {
                    return true;
                }
            }
        }
    }
    
    // Check function attributes
    if (F.hasFnAttribute("annotate")) {
        Attribute Attr = F.getFnAttribute("annotate");
        if (Attr.isStringAttribute()) {
            StringRef Value = Attr.getValueAsString();
            return Value.startswith(AttrName);
        }
    }
    
    // Check instructions
    if (!F.isDeclaration()) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (MDNode *MD = I.getMetadata("llvm.ptr.annotation")) {
                    for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                        if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                            if (Str->getString().startswith(AttrName)) {
                                return true;
                            }
                        }
                    }
                }
                break;  // Only check first instruction
            }
            break;
        }
    }
    
    return false;
}

/**
 * Extract annotation parameter value
 */
std::string extractAnnotationParam(Function &F, StringRef AttrName) {
    // Check metadata first
    if (MDNode *MD = F.getMetadata("llvm.ptr.annotation")) {
        for (unsigned i = 0; i < MD->getNumOperands(); i++) {
            if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                StringRef Value = Str->getString();
                if (Value.startswith(AttrName + "=")) {
                    return Value.substr(AttrName.size() + 1).str();
                }
            }
        }
    }
    
    // Check function attributes
    if (F.hasFnAttribute("annotate")) {
        Attribute Attr = F.getFnAttribute("annotate");
        if (Attr.isStringAttribute()) {
            StringRef Value = Attr.getValueAsString();
            if (Value.startswith(AttrName + "=")) {
                return Value.substr(AttrName.size() + 1).str();
            }
        }
    }
    
    return "";
}

/**
 * Extract DSMIL layer from function
 */
uint8_t extractLayer(Function &F) {
    if (F.hasFnAttribute("dsmil_layer")) {
        Attribute Attr = F.getFnAttribute("dsmil_layer");
        if (Attr.isIntAttribute()) {
            return (uint8_t)Attr.getValueAsInt();
        }
    }
    return 0;
}

/**
 * Extract DSMIL device from function
 */
uint8_t extractDevice(Function &F) {
    if (F.hasFnAttribute("dsmil_device")) {
        Attribute Attr = F.getFnAttribute("dsmil_device");
        if (Attr.isIntAttribute()) {
            return (uint8_t)Attr.getValueAsInt();
        }
    }
    return 0;
}

/**
 * Extract authority tier
 */
uint8_t extractAuthorityTier(Function &F) {
    std::string tier_str = extractAnnotationParam(F, "dsmil.ot_tier");
    if (!tier_str.empty()) {
        return (uint8_t)std::stoi(tier_str);
    }
    
    // Default based on layer
    uint8_t layer = extractLayer(F);
    if (layer <= 1) return 0;  // Safety kernel
    if (layer <= 3) return 1;  // High-impact control
    if (layer <= 6) return 2;  // Optimization
    return 3;  // Analytics
}

/**
 * Collect metrics from module
 */
TelemetryMetrics collectMetrics(Module &Mod) {
    TelemetryMetrics Metrics;
    
    for (Function &F : Mod) {
        if (F.isDeclaration()) continue;
        
        Metrics.total_functions++;
        
        bool isInstrumented = false;
        
        // Check OT annotations
        if (hasAnnotation(F, "dsmil.ot_critical")) {
            Metrics.ot_critical_count++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.ses_gate")) {
            Metrics.ses_gate_count++;
            isInstrumented = true;
        }
        
        // Check generic annotations
        if (hasAnnotation(F, "dsmil.net_io")) {
            Metrics.net_io_count++;
            Metrics.category_counts["net"]++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.crypto")) {
            Metrics.crypto_count++;
            Metrics.category_counts["crypto"]++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.process")) {
            Metrics.process_count++;
            Metrics.category_counts["process"]++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.file")) {
            Metrics.file_count++;
            Metrics.category_counts["file"]++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.untrusted")) {
            Metrics.untrusted_count++;
            Metrics.category_counts["untrusted"]++;
            isInstrumented = true;
        }
        if (hasAnnotation(F, "dsmil.error_handler")) {
            Metrics.error_handler_count++;
            Metrics.category_counts["error"]++;
            isInstrumented = true;
        }
        
        // Count authority tiers
        uint8_t tier = extractAuthorityTier(F);
        switch (tier) {
            case 0: Metrics.tier_0_count++; break;
            case 1: Metrics.tier_1_count++; break;
            case 2: Metrics.tier_2_count++; break;
            case 3: Metrics.tier_3_count++; break;
        }
        
        // Telecom statistics
        std::string telecom_stack = extractAnnotationParam(F, "dsmil.telecom_stack");
        if (!telecom_stack.empty()) {
            Metrics.telecom_stack_count++;
            Metrics.telecom_stacks[telecom_stack]++;
        }
        
        std::string ss7_role = extractAnnotationParam(F, "dsmil.ss7_role");
        if (!ss7_role.empty()) {
            Metrics.ss7_roles[ss7_role]++;
        }
        
        std::string sigtran_role = extractAnnotationParam(F, "dsmil.sigtran_role");
        if (!sigtran_role.empty()) {
            Metrics.sigtran_roles[sigtran_role]++;
        }
        
        std::string telecom_env = extractAnnotationParam(F, "dsmil.telecom_env");
        if (!telecom_env.empty()) {
            Metrics.telecom_envs[telecom_env]++;
        }
        
        // Layer/device distribution
        uint8_t layer = extractLayer(F);
        if (layer > 0) {
            Metrics.layer_counts[layer]++;
        }
        
        uint8_t device = extractDevice(F);
        if (device > 0) {
            Metrics.device_counts[device]++;
        }
        
        if (isInstrumented) {
            Metrics.instrumented_functions++;
        }
    }
    
    // Count safety signals
    for (GlobalVariable &GV : Mod.globals()) {
        if (MDNode *MD = GV.getMetadata("llvm.ptr.annotation")) {
            for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                    StringRef Value = Str->getString();
                    if (Value.startswith("dsmil.safety_signal=")) {
                        Metrics.safety_signal_count++;
                        break;
                    }
                }
            }
        }
    }
    
    return Metrics;
}

/**
 * Generate metrics JSON manifest
 */
void generateMetricsJSON(Module &Mod, const TelemetryMetrics &Metrics,
                        const std::string &OutputDir) {
    std::string ModuleName = Mod.getName().str();
    if (ModuleName.empty()) {
        ModuleName = "unknown";
    }
    
    // Determine output path
    std::string OutputPath;
    if (!OutputDir.empty()) {
        OutputPath = OutputDir + "/" + ModuleName + ".dsmil.metrics.json";
    } else {
        OutputPath = ModuleName + ".dsmil.metrics.json";
    }
    
    // Ensure output directory exists
    if (!OutputDir.empty()) {
        std::error_code EC = llvm::sys::fs::create_directories(OutputDir);
        if (EC) {
            errs() << "Warning: Could not create output directory: " << OutputDir << "\n";
        }
    }
    
    std::ofstream Out(OutputPath);
    if (!Out.is_open()) {
        errs() << "Warning: Could not open metrics file: " << OutputPath << "\n";
        return;
    }
    
    Out << "{\n";
    Out << "  \"module_id\": \"" << ModuleName << "\",\n";
    Out << "  \"mission_profile\": \"" << (MissionProfile.empty() ? "default" : MissionProfile) << "\",\n";
    Out << "  \"metrics\": {\n";
    
    // Overall counts
    Out << "    \"total_functions\": " << Metrics.total_functions << ",\n";
    Out << "    \"instrumented_functions\": " << Metrics.instrumented_functions << ",\n";
    Out << "    \"instrumentation_coverage\": " 
        << (Metrics.total_functions > 0 ? 
            (100.0 * Metrics.instrumented_functions / Metrics.total_functions) : 0.0)
        << ",\n";
    
    // OT-specific counts
    Out << "    \"ot_critical_count\": " << Metrics.ot_critical_count << ",\n";
    Out << "    \"ses_gate_count\": " << Metrics.ses_gate_count << ",\n";
    
    // Generic annotation counts
    Out << "    \"net_io_count\": " << Metrics.net_io_count << ",\n";
    Out << "    \"crypto_count\": " << Metrics.crypto_count << ",\n";
    Out << "    \"process_count\": " << Metrics.process_count << ",\n";
    Out << "    \"file_count\": " << Metrics.file_count << ",\n";
    Out << "    \"untrusted_count\": " << Metrics.untrusted_count << ",\n";
    Out << "    \"error_handler_count\": " << Metrics.error_handler_count << ",\n";
    
    // OT tier distribution
    Out << "    \"authority_tiers\": {\n";
    Out << "      \"tier_0\": " << Metrics.tier_0_count << ",\n";
    Out << "      \"tier_1\": " << Metrics.tier_1_count << ",\n";
    Out << "      \"tier_2\": " << Metrics.tier_2_count << ",\n";
    Out << "      \"tier_3\": " << Metrics.tier_3_count << "\n";
    Out << "    },\n";
    
    // Category distribution
    Out << "    \"categories\": {\n";
    bool first = true;
    for (const auto &Pair : Metrics.category_counts) {
        if (!first) Out << ",\n";
        Out << "      \"" << Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n    },\n";
    
    // Telecom statistics
    Out << "    \"telecom\": {\n";
    Out << "      \"total\": " << Metrics.telecom_stack_count << ",\n";
    Out << "      \"stacks\": {\n";
    first = true;
    for (const auto &Pair : Metrics.telecom_stacks) {
        if (!first) Out << ",\n";
        Out << "        \"" << Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n      },\n";
    Out << "      \"ss7_roles\": {\n";
    first = true;
    for (const auto &Pair : Metrics.ss7_roles) {
        if (!first) Out << ",\n";
        Out << "        \"" << Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n      },\n";
    Out << "      \"sigtran_roles\": {\n";
    first = true;
    for (const auto &Pair : Metrics.sigtran_roles) {
        if (!first) Out << ",\n";
        Out << "        \"" << Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n      },\n";
    Out << "      \"environments\": {\n";
    first = true;
    for (const auto &Pair : Metrics.telecom_envs) {
        if (!first) Out << ",\n";
        Out << "        \"" << Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n      }\n";
    Out << "    },\n";
    
    // Safety signals
    Out << "    \"safety_signals\": " << Metrics.safety_signal_count << ",\n";
    
    // Layer distribution
    Out << "    \"layers\": {\n";
    first = true;
    for (const auto &Pair : Metrics.layer_counts) {
        if (!first) Out << ",\n";
        Out << "      \"" << (int)Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n    },\n";
    
    // Device distribution
    Out << "    \"devices\": {\n";
    first = true;
    for (const auto &Pair : Metrics.device_counts) {
        if (!first) Out << ",\n";
        Out << "      \"" << (int)Pair.first << "\": " << Pair.second;
        first = false;
    }
    Out << "\n    }\n";
    
    Out << "  }\n";
    Out << "}\n";
    Out.close();
    
    outs() << "[DSMIL Metrics] Generated metrics: " << OutputPath << "\n";
}

/**
 * Telemetry Metrics Collection Pass
 */
class DsmilMetricsPass : public PassInfoMixin<DsmilMetricsPass> {
public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        outs() << "[DSMIL Metrics] Collecting metrics for module: " << Mod.getName() << "\n";
        
        TelemetryMetrics Metrics = collectMetrics(Mod);
        
        outs() << "  Total Functions: " << Metrics.total_functions << "\n";
        outs() << "  Instrumented: " << Metrics.instrumented_functions << "\n";
        outs() << "  OT-Critical: " << Metrics.ot_critical_count << "\n";
        outs() << "  Generic Annotations: " 
               << (Metrics.net_io_count + Metrics.crypto_count + Metrics.process_count +
                   Metrics.file_count + Metrics.untrusted_count + Metrics.error_handler_count)
               << "\n";
        
        generateMetricsJSON(Mod, Metrics, MetricsOutputDir);
        
        return PreservedAnalyses::all();  // We don't modify the IR
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DsmilMetricsPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsmil-metrics") {
                        MPM.addPass(DsmilMetricsPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
