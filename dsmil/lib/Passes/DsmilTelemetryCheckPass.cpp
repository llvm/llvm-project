/**
 * @file DsmilTelemetryCheckPass.cpp
 * @brief DSLLVM Telemetry Enforcement Pass (v1.3)
 *
 * This pass enforces telemetry requirements for safety-critical and
 * mission-critical functions. Prevents "dark functions" with zero
 * forensic trail by requiring telemetry calls.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-telemetry-check"

using namespace llvm;

// Command-line options
static cl::opt<std::string> TelemetryCheckMode(
    "dsmil-telemetry-check-mode",
    cl::desc("Telemetry enforcement mode (enforce, warn, disabled)"),
    cl::init("enforce"));

static cl::opt<bool> TelemetryCheckCallGraph(
    "dsmil-telemetry-check-callgraph",
    cl::desc("Check entire call graph for telemetry (default: true)"),
    cl::init(true));

namespace {

/**
 * Telemetry requirement level
 */
enum TelemetryRequirement {
  TELEM_NONE = 0,       /**< No requirement */
  TELEM_BASIC = 1,      /**< At least one telemetry call (safety_critical) */
  TELEM_COMPREHENSIVE = 2  /**< Comprehensive telemetry (mission_critical) */
};

/**
 * Known telemetry functions
 */
const std::set<std::string> TELEMETRY_FUNCTIONS = {
  "dsmil_counter_inc",
  "dsmil_counter_add",
  "dsmil_event_log",
  "dsmil_event_log_severity",
  "dsmil_event_log_msg",
  "dsmil_event_log_structured",
  "dsmil_perf_start",
  "dsmil_perf_end",
  "dsmil_perf_latency",
  "dsmil_perf_throughput",
  "dsmil_forensic_checkpoint",
  "dsmil_forensic_security_event"
};

const std::set<std::string> COUNTER_FUNCTIONS = {
  "dsmil_counter_inc",
  "dsmil_counter_add"
};

const std::set<std::string> EVENT_FUNCTIONS = {
  "dsmil_event_log",
  "dsmil_event_log_severity",
  "dsmil_event_log_msg",
  "dsmil_event_log_structured"
};

/**
 * Telemetry Check Pass
 */
class DsmilTelemetryCheckPass : public PassInfoMixin<DsmilTelemetryCheckPass> {
private:
  std::string EnforcementMode;
  bool CheckCallGraph;

  // Analysis results
  std::map<Function*, std::set<std::string>> FunctionTelemetry;
  std::set<Function*> TelemetryProviders;

  /**
   * Get telemetry requirement for function
   */
  TelemetryRequirement getTelemetryRequirement(Function &F) {
    // Check for mission_critical attribute
    if (F.hasFnAttribute("dsmil_mission_critical")) {
      return TELEM_COMPREHENSIVE;
    }

    // Check for safety_critical attribute
    if (F.hasFnAttribute("dsmil_safety_critical")) {
      return TELEM_BASIC;
    }

    return TELEM_NONE;
  }

  /**
   * Check if function is a telemetry provider
   */
  bool isTelemetryProvider(Function &F) {
    return F.hasFnAttribute("dsmil_telemetry");
  }

  /**
   * Find all direct telemetry calls in function
   */
  void findDirectTelemetryCalls(Function &F, std::set<std::string> &Calls) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee) continue;

          StringRef CalleeName = Callee->getName();
          if (TELEMETRY_FUNCTIONS.count(CalleeName.str())) {
            Calls.insert(CalleeName.str());
          }
        }
      }
    }
  }

  /**
   * Find telemetry calls in call graph (transitive)
   */
  void findTransitiveTelemetryCalls(Function &F,
                                     std::set<std::string> &Calls,
                                     std::set<Function*> &Visited) {
    // Avoid infinite recursion
    if (Visited.count(&F)) return;
    Visited.insert(&F);

    // Check direct calls
    findDirectTelemetryCalls(F, Calls);

    // Check callees
    if (CheckCallGraph) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (CallInst *CI = dyn_cast<CallInst>(&I)) {
            Function *Callee = CI->getCalledFunction();
            if (!Callee || Callee->isDeclaration()) continue;

            // Recursively check callee
            findTransitiveTelemetryCalls(*Callee, Calls, Visited);
          }
        }
      }
    }
  }

  /**
   * Analyze telemetry calls in module
   */
  void analyzeTelemetry(Module &M) {
    // Identify telemetry providers
    for (Function &F : M) {
      if (isTelemetryProvider(F)) {
        TelemetryProviders.insert(&F);
      }
    }

    // Analyze each function
    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      if (TelemetryProviders.count(&F)) continue;  // Skip providers

      std::set<std::string> Calls;
      std::set<Function*> Visited;
      findTransitiveTelemetryCalls(F, Calls, Visited);

      FunctionTelemetry[&F] = Calls;
    }
  }

  /**
   * Validate function telemetry against requirements
   */
  bool validateFunction(Function &F, std::vector<std::string> &Violations) {
    TelemetryRequirement Req = getTelemetryRequirement(F);
    if (Req == TELEM_NONE) return true;  // No requirement

    std::set<std::string> &Calls = FunctionTelemetry[&F];

    if (Req == TELEM_BASIC) {
      // Requires at least one telemetry call
      if (Calls.empty()) {
        Violations.push_back(
          "Function '" + F.getName().str() +
          "' is marked dsmil_safety_critical but has no telemetry calls");
        return false;
      }

      LLVM_DEBUG(dbgs() << "[Telemetry Check] '" << F.getName()
                        << "' has " << Calls.size() << " telemetry call(s)\n");
      return true;
    }

    if (Req == TELEM_COMPREHENSIVE) {
      // Requires both counter and event telemetry
      bool HasCounter = false;
      bool HasEvent = false;

      for (const auto &Call : Calls) {
        if (COUNTER_FUNCTIONS.count(Call)) HasCounter = true;
        if (EVENT_FUNCTIONS.count(Call)) HasEvent = true;
      }

      if (!HasCounter) {
        Violations.push_back(
          "Function '" + F.getName().str() +
          "' is marked dsmil_mission_critical but has no counter telemetry " +
          "(dsmil_counter_inc/add required)");
      }

      if (!HasEvent) {
        Violations.push_back(
          "Function '" + F.getName().str() +
          "' is marked dsmil_mission_critical but has no event telemetry " +
          "(dsmil_event_log* required)");
      }

      if (Calls.empty()) {
        Violations.push_back(
          "Function '" + F.getName().str() +
          "' is marked dsmil_mission_critical but has no telemetry calls");
      }

      return HasCounter && HasEvent;
    }

    return true;
  }

  /**
   * Check error path coverage (mission_critical only)
   */
  bool checkErrorPathCoverage(Function &F, std::vector<std::string> &Violations) {
    TelemetryRequirement Req = getTelemetryRequirement(F);
    if (Req != TELEM_COMPREHENSIVE) return true;

    // Simple heuristic: check that returns with error codes have telemetry
    // This is a simplified check; full implementation would do dataflow analysis

    Type *RetTy = F.getReturnType();
    if (!RetTy->isIntegerTy()) return true;  // Not an error-returning function

    bool HasErrorReturn = false;
    bool AllErrorPathsLogged = true;

    for (BasicBlock &BB : F) {
      ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!RI) continue;

      Value *RetVal = RI->getReturnValue();
      if (!RetVal) continue;

      // Check if this looks like an error return (heuristic: < 0)
      if (ConstantInt *CI = dyn_cast<ConstantInt>(RetVal)) {
        if (CI->getSExtValue() < 0) {
          HasErrorReturn = true;

          // Check if this BB or its predecessors have event logging
          bool HasLog = false;
          for (Instruction &I : BB) {
            if (CallInst *Call = dyn_cast<CallInst>(&I)) {
              Function *Callee = Call->getCalledFunction();
              if (Callee && EVENT_FUNCTIONS.count(Callee->getName().str())) {
                HasLog = true;
                break;
              }
            }
          }

          if (!HasLog) {
            AllErrorPathsLogged = false;
          }
        }
      }
    }

    if (HasErrorReturn && !AllErrorPathsLogged) {
      Violations.push_back(
        "Function '" + F.getName().str() +
        "' is marked dsmil_mission_critical but some error paths lack telemetry");
      return false;
    }

    return true;
  }

public:
  DsmilTelemetryCheckPass()
    : EnforcementMode(TelemetryCheckMode),
      CheckCallGraph(TelemetryCheckCallGraph) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (EnforcementMode == "disabled") {
      LLVM_DEBUG(dbgs() << "[Telemetry Check] Disabled\n");
      return PreservedAnalyses::all();
    }

    outs() << "[DSMIL Telemetry Check] Analyzing telemetry requirements...\n";

    // Analyze all telemetry calls
    analyzeTelemetry(M);

    // Count functions with requirements
    int SafetyCriticalCount = 0;
    int MissionCriticalCount = 0;
    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      TelemetryRequirement Req = getTelemetryRequirement(F);
      if (Req == TELEM_BASIC) SafetyCriticalCount++;
      if (Req == TELEM_COMPREHENSIVE) MissionCriticalCount++;
    }

    outs() << "  Safety-Critical Functions: " << SafetyCriticalCount << "\n";
    outs() << "  Mission-Critical Functions: " << MissionCriticalCount << "\n";
    outs() << "  Telemetry Providers: " << TelemetryProviders.size() << "\n";

    // Validate all functions
    std::vector<std::string> AllViolations;
    int ViolationCount = 0;

    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      if (TelemetryProviders.count(&F)) continue;

      std::vector<std::string> FuncViolations;
      bool Valid = validateFunction(F, FuncViolations);

      // Check error path coverage for mission_critical
      Valid = checkErrorPathCoverage(F, FuncViolations) && Valid;

      if (!Valid) {
        ViolationCount++;
        AllViolations.insert(AllViolations.end(),
                           FuncViolations.begin(),
                           FuncViolations.end());
      }
    }

    // Report violations
    if (!AllViolations.empty()) {
      errs() << "\n[DSMIL Telemetry Check] Telemetry Violations ("
             << ViolationCount << " functions):\n";
      for (const auto &V : AllViolations) {
        errs() << "  ERROR: " << V << "\n";
      }
      errs() << "\n";

      errs() << "Hint: Add telemetry calls to satisfy requirements:\n";
      errs() << "  - Safety-critical: At least one telemetry call\n";
      errs() << "    Example: dsmil_counter_inc(\"function_calls\");\n";
      errs() << "  - Mission-critical: Both counter AND event telemetry\n";
      errs() << "    Example: dsmil_counter_inc(\"calls\");\n";
      errs() << "             dsmil_event_log(\"operation_start\");\n";
      errs() << "\nSee: dsmil/include/dsmil_telemetry.h\n";

      if (EnforcementMode == "enforce") {
        errs() << "\n[DSMIL Telemetry Check] FATAL: Telemetry violations detected\n";
        report_fatal_error("Telemetry enforcement failure");
      } else {
        errs() << "\n[DSMIL Telemetry Check] WARNING: Violations detected but enforcement mode is 'warn'\n";
      }
    } else {
      if (SafetyCriticalCount > 0 || MissionCriticalCount > 0) {
        outs() << "[DSMIL Telemetry Check] ✓ All functions satisfy telemetry requirements\n";
      } else {
        outs() << "[DSMIL Telemetry Check] No telemetry requirements found\n";
      }
    }

    // Add module-level metadata
    LLVMContext &Ctx = M.getContext();
    M.setModuleFlag(Module::Warning, "dsmil.telemetry_safety_critical_count",
                   MDString::get(Ctx, std::to_string(SafetyCriticalCount)));
    M.setModuleFlag(Module::Warning, "dsmil.telemetry_mission_critical_count",
                   MDString::get(Ctx, std::to_string(MissionCriticalCount)));

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return false; }
};

} // anonymous namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilTelemetryCheckPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-telemetry-check") {
            MPM.addPass(DsmilTelemetryCheckPass());
            return true;
          }
          return false;
        });
    }
  };
}
