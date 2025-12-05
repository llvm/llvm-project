/**
 * @file DsmilFuzzExportPass.cpp
 * @brief DSLLVM Auto-Generated Fuzz Harness Export Pass (v1.3)
 *
 * This pass automatically identifies untrusted input functions and exports
 * fuzz harness specifications that can be consumed by fuzzing engines
 * (libFuzzer, AFL++, etc.) or AI-assisted harness generators.
 *
 * Key Features:
 * - Detects functions with dsmil_untrusted_input attribute
 * - Analyzes parameter types and domains
 * - Computes Layer 8 Security AI risk scores
 * - Exports *.dsmilfuzz.json sidecar files
 * - Integrates with L7 LLM for harness code generation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-fuzz-export"

using namespace llvm;

// Command-line options
static cl::opt<std::string> FuzzExportPath(
    "dsmil-fuzz-export-path",
    cl::desc("Output directory for .dsmilfuzz.json files"),
    cl::init("."));

static cl::opt<bool> FuzzExportEnabled(
    "fdsmil-fuzz-export",
    cl::desc("Enable automatic fuzz harness export"),
    cl::init(true));

static cl::opt<float> FuzzRiskThreshold(
    "dsmil-fuzz-risk-threshold",
    cl::desc("Minimum risk score to export fuzz target (0.0-1.0)"),
    cl::init(0.3));

static cl::opt<bool> FuzzL7LLMIntegration(
    "dsmil-fuzz-l7-llm",
    cl::desc("Enable Layer 7 LLM harness generation"),
    cl::init(false));

namespace {

/**
 * Fuzz target parameter descriptor
 */
struct FuzzParameter {
  std::string name;
  std::string type;
  std::optional<std::string> length_ref;  // For buffers: which param is the length
  std::optional<int> min_value;
  std::optional<int> max_value;
  bool is_untrusted;
};

/**
 * Fuzz target descriptor
 */
struct FuzzTarget {
  std::string function_name;
  std::vector<std::string> untrusted_params;
  std::map<std::string, FuzzParameter> parameter_domains;
  float l8_risk_score;
  std::string priority;  // "high", "medium", "low"
  std::optional<int> layer;
  std::optional<int> device;
  std::optional<std::string> stage;
};

/**
 * Auto-Generated Fuzz Harness Export Pass
 */
class DsmilFuzzExportPass : public PassInfoMixin<DsmilFuzzExportPass> {
private:
  std::vector<FuzzTarget> Targets;
  std::string OutputPath;

  /**
   * Check if function has untrusted input attribute
   */
  bool hasUntrustedInput(Function &F) {
    return F.hasFnAttribute("dsmil_untrusted_input");
  }

  /**
   * Extract attribute value from function
   */
  std::optional<std::string> getAttributeValue(Function &F, StringRef AttrName) {
    if (Attribute Attr = F.getFnAttribute(AttrName); Attr.isStringAttribute()) {
      return Attr.getValueAsString().str();
    }
    return std::nullopt;
  }

  /**
   * Extract integer attribute value
   */
  std::optional<int> getIntAttributeValue(Function &F, StringRef AttrName) {
    if (Attribute Attr = F.getFnAttribute(AttrName); Attr.isStringAttribute()) {
      StringRef Val = Attr.getValueAsString();
      int Result;
      if (!Val.getAsInteger(10, Result))
        return Result;
    }
    return std::nullopt;
  }

  /**
   * Convert LLVM type to human-readable string
   */
  std::string typeToString(Type *Ty) {
    if (Ty->isIntegerTy()) {
      return "int" + std::to_string(Ty->getIntegerBitWidth()) + "_t";
    } else if (Ty->isFloatTy()) {
      return "float";
    } else if (Ty->isDoubleTy()) {
      return "double";
    } else if (Ty->isPointerTy()) {
      return "pointer";
    } else if (Ty->isStructTy()) {
      return "struct";
    } else if (Ty->isArrayTy()) {
      return "array";
    }
    return "unknown";
  }

  /**
   * Analyze function parameters to determine fuzz domains
   */
  void analyzeParameters(Function &F, FuzzTarget &Target) {
    int ParamIdx = 0;
    std::string LengthParam;

    for (Argument &Arg : F.args()) {
      FuzzParameter Param;
      Param.name = Arg.getName().str();
      if (Param.name.empty()) {
        Param.name = "arg" + std::to_string(ParamIdx);
      }

      Type *ArgTy = Arg.getType();
      Param.type = typeToString(ArgTy);
      Param.is_untrusted = true;  // All params in untrusted input function

      // Detect length parameters
      if (Param.name.find("len") != std::string::npos ||
          Param.name.find("size") != std::string::npos ||
          Param.name.find("count") != std::string::npos) {
        LengthParam = Param.name;
      }

      // Set reasonable defaults for numeric types
      if (ArgTy->isIntegerTy()) {
        if (ArgTy->getIntegerBitWidth() <= 32) {
          Param.min_value = 0;
          Param.max_value = (1 << 16) - 1;  // 64KB max for sizes
        } else {
          Param.min_value = 0;
          Param.max_value = (1 << 20) - 1;  // 1MB max for 64-bit sizes
        }
      }

      Target.parameter_domains[Param.name] = Param;
      Target.untrusted_params.push_back(Param.name);
      ParamIdx++;
    }

    // Link buffer parameters to their length parameters
    if (!LengthParam.empty()) {
      for (auto &Entry : Target.parameter_domains) {
        FuzzParameter &Param = Entry.second;
        if (Param.type == "bytes" && !Param.length_ref.has_value()) {
          Param.length_ref = LengthParam;
        }
      }
    }
  }

  /**
   * Compute Layer 8 Security AI risk score
   *
   * This is a simplified heuristic. In production, this would:
   * 1. Extract function IR features
   * 2. Invoke Layer 8 Security AI model (ONNX on Device 80)
   * 3. Return ML-predicted vulnerability risk
   */
  float computeL8RiskScore(Function &F) {
    float risk = 0.0f;

    // Heuristic factors:

    // 1. Function name patterns
    StringRef Name = F.getName();
    if (Name.contains("parse") || Name.contains("decode")) risk += 0.3f;
    if (Name.contains("network") || Name.contains("socket")) risk += 0.3f;
    if (Name.contains("file") || Name.contains("read")) risk += 0.2f;
    if (Name.contains("crypto") || Name.contains("hash")) risk += 0.1f;

    // 2. Parameter complexity (more params = more attack surface)
    size_t ParamCount = F.arg_size();
    if (ParamCount >= 5) risk += 0.2f;
    else if (ParamCount >= 3) risk += 0.1f;

    // 3. Pointer parameters (potential buffer overflows)
    int PointerParams = 0;
    for (Argument &Arg : F.args()) {
      if (Arg.getType()->isPointerTy()) PointerParams++;
    }
    if (PointerParams >= 2) risk += 0.2f;

    // 4. Layer assignment (lower layers = more privilege)
    if (auto Layer = getIntAttributeValue(F, "dsmil_layer")) {
      if (*Layer <= 3) risk += 0.2f;  // Kernel/crypto layers
      else if (*Layer <= 5) risk += 0.1f;  // System services
    }

    // Cap at 1.0
    return risk > 1.0f ? 1.0f : risk;
  }

  /**
   * Determine priority based on risk score
   */
  std::string riskToPriority(float risk) {
    if (risk >= 0.7) return "high";
    if (risk >= 0.4) return "medium";
    return "low";
  }

  /**
   * Export fuzz target to JSON
   */
  void exportFuzzTarget(Module &M, const FuzzTarget &Target) {
    std::string Filename = OutputPath + "/" + M.getName().str() + ".dsmilfuzz.json";

    std::error_code EC;
    raw_fd_ostream OutFile(Filename, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "[DSMIL Fuzz Export] ERROR: Failed to open " << Filename
             << ": " << EC.message() << "\n";
      return;
    }

    // Build JSON structure
    json::Object Root;
    Root["schema"] = "dsmil-fuzz-v1";
    Root["version"] = "1.3.0";
    Root["binary"] = M.getName().str();
    Root["generated_at"] = "2026-01-15T14:30:00Z";  // TODO: Real timestamp

    // Fuzz targets array
    json::Array TargetsArray;
    json::Object TargetObj;
    TargetObj["function"] = Target.function_name;
    TargetObj["l8_risk_score"] = Target.l8_risk_score;
    TargetObj["priority"] = Target.priority;

    // Untrusted parameters
    json::Array UntrustedParams;
    for (const auto &Param : Target.untrusted_params) {
      UntrustedParams.push_back(Param);
    }
    TargetObj["untrusted_params"] = std::move(UntrustedParams);

    // Parameter domains
    json::Object ParamDomains;
    for (const auto &Entry : Target.parameter_domains) {
      const FuzzParameter &Param = Entry.second;
      json::Object ParamObj;
      ParamObj["type"] = Param.type;
      if (Param.length_ref) ParamObj["length_ref"] = *Param.length_ref;
      if (Param.min_value) ParamObj["min"] = *Param.min_value;
      if (Param.max_value) ParamObj["max"] = *Param.max_value;
      ParamDomains[Param.name] = std::move(ParamObj);
    }
    TargetObj["parameter_domains"] = std::move(ParamDomains);

    // Metadata
    if (Target.layer) TargetObj["layer"] = *Target.layer;
    if (Target.device) TargetObj["device"] = *Target.device;
    if (Target.stage) TargetObj["stage"] = *Target.stage;

    TargetsArray.push_back(std::move(TargetObj));
    Root["fuzz_targets"] = std::move(TargetsArray);

    // L7 LLM integration metadata
    if (FuzzL7LLMIntegration) {
      json::Object L7Meta;
      L7Meta["enabled"] = true;
      L7Meta["request_harness_generation"] = true;
      L7Meta["target_fuzzer"] = "libFuzzer";
      L7Meta["output_language"] = "C++";
      Root["l7_llm_integration"] = std::move(L7Meta);
    }

    // Write JSON
    json::Value JsonVal(std::move(Root));
    OutFile << formatv("{0:2}", JsonVal) << "\n";
    OutFile.close();

    outs() << "[DSMIL Fuzz Export] ✓ Exported fuzz target: " << Filename << "\n";
    outs() << "  Function: " << Target.function_name << "\n";
    outs() << "  Risk Score: " << format("%.2f", Target.l8_risk_score) << " (" << Target.priority << ")\n";
    outs() << "  Parameters: " << Target.untrusted_params.size() << "\n";
  }

public:
  DsmilFuzzExportPass() : OutputPath(FuzzExportPath) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (!FuzzExportEnabled) {
      LLVM_DEBUG(dbgs() << "[DSMIL Fuzz Export] Disabled, skipping\n");
      return PreservedAnalyses::all();
    }

    outs() << "[DSMIL Fuzz Export] Analyzing untrusted input functions...\n";

    // Identify all fuzz targets
    Targets.clear();
    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      if (!hasUntrustedInput(F)) continue;

      FuzzTarget Target;
      Target.function_name = F.getName().str();

      // Extract DSMIL metadata
      Target.layer = getIntAttributeValue(F, "dsmil_layer");
      Target.device = getIntAttributeValue(F, "dsmil_device");
      Target.stage = getAttributeValue(F, "dsmil_stage");

      // Analyze parameters
      analyzeParameters(F, Target);

      // Compute risk score
      Target.l8_risk_score = computeL8RiskScore(F);
      Target.priority = riskToPriority(Target.l8_risk_score);

      // Filter by risk threshold
      if (Target.l8_risk_score < FuzzRiskThreshold) {
        LLVM_DEBUG(dbgs() << "[DSMIL Fuzz Export] Skipping '" << Target.function_name
                          << "' (risk " << Target.l8_risk_score << " < threshold "
                          << FuzzRiskThreshold << ")\n");
        continue;
      }

      Targets.push_back(Target);
    }

    if (Targets.empty()) {
      outs() << "[DSMIL Fuzz Export] No untrusted input functions found\n";
      return PreservedAnalyses::all();
    }

    outs() << "[DSMIL Fuzz Export] Found " << Targets.size() << " fuzz target(s)\n";

    // Export each target
    for (const auto &Target : Targets) {
      exportFuzzTarget(M, Target);
    }

    // Add module-level metadata
    LLVMContext &Ctx = M.getContext();
    M.setModuleFlag(Module::Warning, "dsmil.fuzz_targets_exported",
                   MDString::get(Ctx, std::to_string(Targets.size())));

    if (FuzzL7LLMIntegration) {
      outs() << "\n[DSMIL Fuzz Export] Layer 7 LLM Integration Enabled\n";
      outs() << "  → Run: dsmil-fuzz-gen " << M.getName().str() << ".dsmilfuzz.json\n";
      outs() << "  → This will generate libFuzzer harnesses using L7 LLM\n";
    }

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return false; }
};

} // anonymous namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilFuzzExportPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-fuzz-export") {
            MPM.addPass(DsmilFuzzExportPass());
            return true;
          }
          return false;
        });
    }
  };
}
