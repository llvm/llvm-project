/**
 * @file DsmilBlueRedPass.cpp
 * @brief DSLLVM Blue vs Red Scenario Simulation Pass (v1.4 - Feature 2.3)
 *
 * This pass implements dual-build instrumentation for adversarial testing.
 * Blue builds (production) are normal; Red builds (testing) include extra
 * instrumentation to simulate attack scenarios and map blast radius.
 *
 * Red builds are NEVER deployed to production and must be confined to
 * isolated test environments.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-blue-red"

using namespace llvm;

// Command-line options
static cl::opt<std::string> BuildRole(
    "fdsmil-role",
    cl::desc("Build role: blue (defender) or red (attacker stress-test)"),
    cl::init("blue"));

static cl::opt<bool> RedInstrumentation(
    "dsmil-red-instrument",
    cl::desc("Enable red team instrumentation"),
    cl::init(true));

static cl::opt<bool> AttackSurfaceMapping(
    "dsmil-red-attack-surface",
    cl::desc("Enable attack surface mapping in red builds"),
    cl::init(true));

static cl::opt<bool> VulnInjection(
    "dsmil-red-vuln-inject",
    cl::desc("Enable vulnerability injection points in red builds"),
    cl::init(false));

static cl::opt<std::string> RedOutputPath(
    "dsmil-red-output",
    cl::desc("Output path for red build analysis report"),
    cl::init("red-analysis.json"));

namespace {

/**
 * Build role enumeration
 */
enum BuildRoleEnum {
  ROLE_BLUE = 0,  // Production/defender build
  ROLE_RED = 1    // Testing/attacker stress-test build
};

/**
 * Attack surface classification
 */
struct AttackSurfaceInfo {
  std::string function_name;
  std::string location;
  uint32_t layer;
  uint32_t device;
  bool has_untrusted_input;
  std::vector<std::string> entry_points;
  std::vector<std::string> vulnerabilities;
  uint32_t blast_radius_score;  // 0-100
};

/**
 * Red team hook information
 */
struct RedTeamHook {
  std::string hook_name;
  std::string function_name;
  std::string hook_type;  // "injection_point", "bypass", "exploit"
  uint32_t line_number;
};

/**
 * Blue vs Red Simulation Pass
 */
class DsmilBlueRedPass : public PassInfoMixin<DsmilBlueRedPass> {
private:
  std::string Role;
  bool IsRedBuild;
  bool Instrument;
  bool MapAttackSurface;
  bool InjectVulns;
  std::string OutputPath;

  // Analysis data
  std::vector<AttackSurfaceInfo> AttackSurfaces;
  std::vector<RedTeamHook> RedHooks;
  std::set<std::string> BlastRadiusFunctions;

  // Statistics
  unsigned RedHooksInserted = 0;
  unsigned AttackSurfacesMapped = 0;
  unsigned VulnInjectionsAdded = 0;
  unsigned BlastRadiusTracked = 0;

  /**
   * Determine build role from CLI or attributes
   */
  BuildRoleEnum getBuildRole(Module &M) {
    // Check CLI flag first
    if (Role == "red")
      return ROLE_RED;

    // Check module-level attribute
    if (M.getModuleFlag("dsmil.build_role")) {
      auto *MD = cast<MDString>(M.getModuleFlag("dsmil.build_role"));
      if (MD->getString() == "red")
        return ROLE_RED;
    }

    return ROLE_BLUE;
  }

  /**
   * Check if function has red team hook attribute
   */
  bool hasRedTeamHook(Function &F, std::string &HookName) {
    if (F.hasFnAttribute("dsmil_red_team_hook")) {
      Attribute Attr = F.getFnAttribute("dsmil_red_team_hook");
      HookName = Attr.getValueAsString().str();
      return true;
    }
    return false;
  }

  /**
   * Check if function is attack surface
   */
  bool isAttackSurface(Function &F) {
    return F.hasFnAttribute("dsmil_attack_surface") ||
           F.hasFnAttribute("dsmil_untrusted_input");
  }

  /**
   * Check if function has vulnerability injection point
   */
  bool hasVulnInject(Function &F, std::string &VulnType) {
    if (F.hasFnAttribute("dsmil_vuln_inject")) {
      Attribute Attr = F.getFnAttribute("dsmil_vuln_inject");
      VulnType = Attr.getValueAsString().str();
      return true;
    }
    return false;
  }

  /**
   * Check if function has blast radius tracking
   */
  bool hasBlastRadius(Function &F) {
    return F.hasFnAttribute("dsmil_blast_radius");
  }

  /**
   * Get layer/device from function attributes
   */
  void getLayerDevice(Function &F, uint32_t &Layer, uint32_t &Device) {
    Layer = 0;
    Device = 0;

    if (F.hasFnAttribute("dsmil_layer")) {
      Attribute Attr = F.getFnAttribute("dsmil_layer");
      Layer = std::stoi(Attr.getValueAsString().str());
    }

    if (F.hasFnAttribute("dsmil_device")) {
      Attribute Attr = F.getFnAttribute("dsmil_device");
      Device = std::stoi(Attr.getValueAsString().str());
    }
  }

  /**
   * Insert red team instrumentation at function entry
   */
  bool instrumentRedTeamHook(Function &F, const std::string &HookName) {
    if (!IsRedBuild || !Instrument)
      return false;

    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Insert logging call at function entry
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> Builder(&Entry, Entry.getFirstInsertionPt());

    // Create call to dsmil_red_log(hook_name, function_name)
    auto *I8Ptr = PointerType::get(Type::getInt8Ty(Ctx), 0);
    FunctionCallee RedLogFunc = M->getOrInsertFunction(
        "dsmil_red_log", Type::getVoidTy(Ctx), I8Ptr, I8Ptr);

    Value *HookNameStr = Builder.CreateGlobalStringPtr(HookName);
    Value *FuncNameStr = Builder.CreateGlobalStringPtr(F.getName());

    Builder.CreateCall(RedLogFunc, {HookNameStr, FuncNameStr});

    RedHooksInserted++;

    // Record hook
    RedTeamHook Hook;
    Hook.hook_name = HookName;
    Hook.function_name = F.getName().str();
    Hook.hook_type = "instrumentation";
    Hook.line_number = 0;  // TODO: Get from debug info
    RedHooks.push_back(Hook);

    return true;
  }

  /**
   * Map attack surface for function
   */
  bool mapAttackSurface(Function &F) {
    if (!IsRedBuild || !MapAttackSurface)
      return false;

    AttackSurfaceInfo Info;
    Info.function_name = F.getName().str();
    Info.location = ""; // TODO: Get from debug info

    getLayerDevice(F, Info.layer, Info.device);

    Info.has_untrusted_input = F.hasFnAttribute("dsmil_untrusted_input");

    // Calculate blast radius score (simplified)
    uint32_t Score = 0;
    if (Info.layer >= 7) Score += 30;  // High layer = higher impact
    if (Info.has_untrusted_input) Score += 40;  // Untrusted input = high risk
    if (F.hasFnAttribute("dsmil_safety_critical")) Score += 20;
    if (F.hasFnAttribute("dsmil_mission_critical")) Score += 30;

    Info.blast_radius_score = std::min(Score, 100u);

    AttackSurfaces.push_back(Info);
    AttackSurfacesMapped++;

    return true;
  }

  /**
   * Add vulnerability injection instrumentation
   */
  bool addVulnInjection(Function &F, const std::string &VulnType) {
    if (!IsRedBuild || !InjectVulns)
      return false;

    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Insert scenario check at function entry
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> Builder(&Entry, Entry.getFirstInsertionPt());

    // Create call to dsmil_red_scenario(vuln_type)
    auto *I8Ptr = PointerType::get(Type::getInt8Ty(Ctx), 0);
    FunctionCallee ScenarioFunc = M->getOrInsertFunction(
        "dsmil_red_scenario", Type::getInt1Ty(Ctx), I8Ptr);

    Value *VulnTypeStr = Builder.CreateGlobalStringPtr(VulnType);
    Value *ShouldInject = Builder.CreateCall(ScenarioFunc, {VulnTypeStr});

    // Create conditional instrumentation (simplified)
    // In real implementation, this would inject specific vulnerability patterns

    VulnInjectionsAdded++;

    return true;
  }

  /**
   * Track blast radius function
   */
  bool trackBlastRadius(Function &F) {
    if (!IsRedBuild)
      return false;

    BlastRadiusFunctions.insert(F.getName().str());
    BlastRadiusTracked++;

    return true;
  }

  /**
   * Add metadata to mark red build
   */
  void addRedBuildMetadata(Module &M) {
    if (!IsRedBuild)
      return;

    LLVMContext &Ctx = M.getContext();

    // Add module flag
    M.addModuleFlag(Module::Warning, "dsmil.build_role",
                   MDString::get(Ctx, "red"));

    // Add warning metadata
    SmallVector<Metadata *, 2> WarningMD;
    WarningMD.push_back(MDString::get(Ctx, "dsmil.red_build.warning"));
    WarningMD.push_back(MDString::get(Ctx,
        "RED BUILD - FOR TESTING ONLY - NEVER DEPLOY TO PRODUCTION"));

    MDNode *Warning = MDNode::get(Ctx, WarningMD);
    M.addModuleFlag(Module::Warning, "dsmil.red_build", Warning);
  }

  /**
   * Generate red build analysis report
   */
  void generateAnalysisReport(Module &M) {
    if (!IsRedBuild)
      return;

    using namespace llvm::json;

    // Create JSON report
    Object Report;
    Report["schema"] = "dsmil-red-analysis-v1";
    Report["module"] = M.getName().str();
    Report["build_role"] = "red";

    // Statistics
    Object Stats;
    Stats["red_hooks_inserted"] = RedHooksInserted;
    Stats["attack_surfaces_mapped"] = AttackSurfacesMapped;
    Stats["vuln_injections_added"] = VulnInjectionsAdded;
    Stats["blast_radius_tracked"] = BlastRadiusTracked;
    Report["statistics"] = std::move(Stats);

    // Attack surfaces
    Array AttackSurfaceArray;
    for (const auto &AS : AttackSurfaces) {
      Object ASObj;
      ASObj["function"] = AS.function_name;
      ASObj["layer"] = AS.layer;
      ASObj["device"] = AS.device;
      ASObj["has_untrusted_input"] = AS.has_untrusted_input;
      ASObj["blast_radius_score"] = AS.blast_radius_score;
      AttackSurfaceArray.push_back(std::move(ASObj));
    }
    Report["attack_surfaces"] = std::move(AttackSurfaceArray);

    // Red team hooks
    Array HooksArray;
    for (const auto &Hook : RedHooks) {
      Object HookObj;
      HookObj["hook_name"] = Hook.hook_name;
      HookObj["function"] = Hook.function_name;
      HookObj["type"] = Hook.hook_type;
      HooksArray.push_back(std::move(HookObj));
    }
    Report["red_hooks"] = std::move(HooksArray);

    // Blast radius functions
    Array BlastArray;
    for (const auto &FName : BlastRadiusFunctions) {
      BlastArray.push_back(FName);
    }
    Report["blast_radius_functions"] = std::move(BlastArray);

    // Write to file
    std::error_code EC;
    raw_fd_ostream OS(OutputPath, EC);
    if (!EC) {
      OS << formatv("{0:2}", json::Value(std::move(Report)));
      OS.close();
    }
  }

public:
  DsmilBlueRedPass()
    : Role(BuildRole.getValue()),
      IsRedBuild(Role == "red"),
      Instrument(RedInstrumentation.getValue()),
      MapAttackSurface(AttackSurfaceMapping.getValue()),
      InjectVulns(VulnInjection.getValue()),
      OutputPath(RedOutputPath.getValue()) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    bool Modified = false;

    LLVM_DEBUG(dbgs() << "[DSMIL Blue/Red] Processing module: "
                      << M.getName() << "\n");
    LLVM_DEBUG(dbgs() << "[DSMIL Blue/Red] Role: " << Role << "\n");

    // Determine build role
    BuildRoleEnum BuildRoleVal = getBuildRole(M);
    IsRedBuild = (BuildRoleVal == ROLE_RED);

    if (IsRedBuild) {
      errs() << "========================================\n";
      errs() << "WARNING: RED TEAM BUILD\n";
      errs() << "FOR TESTING ONLY - NEVER DEPLOY TO PRODUCTION\n";
      errs() << "========================================\n";

      addRedBuildMetadata(M);
      Modified = true;
    }

    // Process functions
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      std::string HookName, VulnType;

      // Red team hooks
      if (hasRedTeamHook(F, HookName)) {
        Modified |= instrumentRedTeamHook(F, HookName);
      }

      // Attack surface mapping
      if (isAttackSurface(F)) {
        Modified |= mapAttackSurface(F);
      }

      // Vulnerability injection
      if (hasVulnInject(F, VulnType)) {
        Modified |= addVulnInjection(F, VulnType);
      }

      // Blast radius tracking
      if (hasBlastRadius(F)) {
        Modified |= trackBlastRadius(F);
      }
    }

    // Generate analysis report
    if (IsRedBuild) {
      generateAnalysisReport(M);

      errs() << "[DSMIL Blue/Red] Red Build Summary:\n";
      errs() << "  Red hooks inserted: " << RedHooksInserted << "\n";
      errs() << "  Attack surfaces mapped: " << AttackSurfacesMapped << "\n";
      errs() << "  Vuln injections added: " << VulnInjectionsAdded << "\n";
      errs() << "  Blast radius tracked: " << BlastRadiusTracked << "\n";
      errs() << "  Analysis report: " << OutputPath << "\n";
    }

    return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // end anonymous namespace

// Register the pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilBlueRedPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-blue-red") {
            MPM.addPass(DsmilBlueRedPass());
            return true;
          }
          return false;
        });
    }
  };
}
