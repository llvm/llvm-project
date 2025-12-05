/**
 * @file DsmilMissionPolicyPass.cpp
 * @brief DSLLVM Mission Profile Policy Enforcement Pass (v1.3)
 *
 * This pass enforces mission profile constraints at compile time.
 * Mission profiles define operational context (border_ops, cyber_defence, etc.)
 * and control compilation behavior, security policies, and runtime constraints.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-mission-policy"

using namespace llvm;

// Command-line options
static cl::opt<std::string> MissionProfile(
    "fdsmil-mission-profile",
    cl::desc("DSMIL mission profile (border_ops, cyber_defence, etc.)"),
    cl::init(""));

static cl::opt<std::string> MissionProfileConfig(
    "fdsmil-mission-profile-config",
    cl::desc("Path to mission-profiles.json"),
    cl::init("/etc/dsmil/mission-profiles.json"));

static cl::opt<std::string> MissionPolicyMode(
    "dsmil-mission-policy-mode",
    cl::desc("Mission policy enforcement mode (enforce, warn, disabled)"),
    cl::init("enforce"));

namespace {

/**
 * Mission profile configuration structure
 */
struct MissionProfileConfig {
  std::string display_name;
  std::string description;
  std::string classification;
  std::string operational_context;
  std::string pipeline;
  std::string ai_mode;
  std::string sandbox_default;
  std::vector<std::string> allow_stages;
  std::vector<std::string> deny_stages;
  bool quantum_export;
  std::string ct_enforcement;
  std::string telemetry_level;
  bool provenance_required;
  std::optional<int> max_deployment_days;
  std::string clearance_floor;
  std::optional<std::vector<int>> device_whitelist;

  // Layer policies: layer_id -> (allowed, roe_required)
  std::map<int, std::pair<bool, std::optional<std::string>>> layer_policies;

  // Compiler flags
  std::vector<std::string> security_flags;
  std::vector<std::string> dsmil_specific_flags;

  // Runtime constraints
  std::optional<int> max_memory_mb;
  std::optional<int> max_cpu_cores;
  bool network_egress_allowed;
  bool filesystem_write_allowed;
};

/**
 * Mission Policy Enforcement Pass
 */
class DsmilMissionPolicyPass : public PassInfoMixin<DsmilMissionPolicyPass> {
private:
  std::string ActiveProfile;
  std::string ConfigPath;
  std::string EnforcementMode;
  MissionProfileConfig CurrentConfig;
  bool ConfigLoaded = false;

  /**
   * Load mission profile configuration from JSON
   */
  bool loadMissionProfile(StringRef ProfileID) {
    auto BufferOrErr = MemoryBuffer::getFile(ConfigPath);
    if (!BufferOrErr) {
      errs() << "[DSMIL Mission Policy] ERROR: Failed to load config from "
             << ConfigPath << ": " << BufferOrErr.getError().message() << "\n";
      return false;
    }

    Expected<json::Value> JsonOrErr = json::parse(BufferOrErr.get()->getBuffer());
    if (!JsonOrErr) {
      errs() << "[DSMIL Mission Policy] ERROR: Failed to parse JSON config: "
             << toString(JsonOrErr.takeError()) << "\n";
      return false;
    }

    const json::Object *Root = JsonOrErr->getAsObject();
    if (!Root) {
      errs() << "[DSMIL Mission Policy] ERROR: Root is not a JSON object\n";
      return false;
    }

    const json::Object *Profiles = Root->getObject("profiles");
    if (!Profiles) {
      errs() << "[DSMIL Mission Policy] ERROR: No 'profiles' section found\n";
      return false;
    }

    const json::Object *Profile = Profiles->getObject(ProfileID);
    if (!Profile) {
      errs() << "[DSMIL Mission Policy] ERROR: Profile '" << ProfileID
             << "' not found. Available profiles: ";
      for (auto &P : *Profiles) {
        errs() << P.first << " ";
      }
      errs() << "\n";
      return false;
    }

    // Parse profile configuration
    CurrentConfig.display_name = Profile->getString("display_name").value_or("");
    CurrentConfig.description = Profile->getString("description").value_or("");
    CurrentConfig.classification = Profile->getString("classification").value_or("");
    CurrentConfig.operational_context = Profile->getString("operational_context").value_or("");
    CurrentConfig.pipeline = Profile->getString("pipeline").value_or("");
    CurrentConfig.ai_mode = Profile->getString("ai_mode").value_or("");
    CurrentConfig.sandbox_default = Profile->getString("sandbox_default").value_or("");
    CurrentConfig.quantum_export = Profile->getBoolean("quantum_export").value_or(false);
    CurrentConfig.ct_enforcement = Profile->getString("ct_enforcement").value_or("");
    CurrentConfig.telemetry_level = Profile->getString("telemetry_level").value_or("");
    CurrentConfig.provenance_required = Profile->getBoolean("provenance_required").value_or(false);
    CurrentConfig.clearance_floor = Profile->getString("clearance_floor").value_or("");
    CurrentConfig.network_egress_allowed = Profile->getBoolean("network_egress_allowed").value_or(true);
    CurrentConfig.filesystem_write_allowed = Profile->getBoolean("filesystem_write_allowed").value_or(true);

    // Parse allow_stages
    if (const json::Array *AllowStages = Profile->getArray("allow_stages")) {
      for (const json::Value &Stage : *AllowStages) {
        if (auto S = Stage.getAsString())
          CurrentConfig.allow_stages.push_back(S->str());
      }
    }

    // Parse deny_stages
    if (const json::Array *DenyStages = Profile->getArray("deny_stages")) {
      for (const json::Value &Stage : *DenyStages) {
        if (auto S = Stage.getAsString())
          CurrentConfig.deny_stages.push_back(S->str());
      }
    }

    // Parse layer policies
    if (const json::Object *LayerPolicy = Profile->getObject("layer_policy")) {
      for (auto &Entry : *LayerPolicy) {
        int LayerID = std::stoi(Entry.first.str());
        const json::Object *Policy = Entry.second.getAsObject();
        if (Policy) {
          bool Allowed = Policy->getBoolean("allowed").value_or(true);
          std::optional<std::string> ROE;
          if (auto ROEVal = Policy->get("roe_required")) {
            if (auto ROEStr = ROEVal->getAsString())
              ROE = ROEStr->str();
          }
          CurrentConfig.layer_policies[LayerID] = {Allowed, ROE};
        }
      }
    }

    // Parse device whitelist
    if (const json::Array *Whitelist = Profile->getArray("device_whitelist")) {
      std::vector<int> Devices;
      for (const json::Value &Dev : *Whitelist) {
        if (auto DevID = Dev.getAsInteger())
          Devices.push_back(*DevID);
      }
      CurrentConfig.device_whitelist = Devices;
    }

    ConfigLoaded = true;

    LLVM_DEBUG(dbgs() << "[DSMIL Mission Policy] Loaded profile '" << ProfileID
                      << "' (" << CurrentConfig.display_name << ")\n");
    LLVM_DEBUG(dbgs() << "  Classification: " << CurrentConfig.classification << "\n");
    LLVM_DEBUG(dbgs() << "  Pipeline: " << CurrentConfig.pipeline << "\n");
    LLVM_DEBUG(dbgs() << "  CT Enforcement: " << CurrentConfig.ct_enforcement << "\n");

    return true;
  }

  /**
   * Extract attribute value from function metadata
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
   * Check if stage is allowed by mission profile
   */
  bool isStageAllowed(StringRef Stage) {
    // If allow_stages is non-empty, stage must be in it
    if (!CurrentConfig.allow_stages.empty()) {
      bool Found = false;
      for (const auto &S : CurrentConfig.allow_stages) {
        if (S == Stage) {
          Found = true;
          break;
        }
      }
      if (!Found)
        return false;
    }

    // Stage must not be in deny_stages
    for (const auto &S : CurrentConfig.deny_stages) {
      if (S == Stage)
        return false;
    }

    return true;
  }

  /**
   * Check if layer is allowed by mission profile
   */
  bool isLayerAllowed(int Layer, std::optional<std::string> &RequiredROE) {
    auto It = CurrentConfig.layer_policies.find(Layer);
    if (It == CurrentConfig.layer_policies.end())
      return true; // No policy = allowed

    RequiredROE = It->second.second;
    return It->second.first;
  }

  /**
   * Check if device is allowed by mission profile
   */
  bool isDeviceAllowed(int DeviceID) {
    if (!CurrentConfig.device_whitelist.has_value())
      return true; // No whitelist = all allowed

    for (int AllowedDev : *CurrentConfig.device_whitelist) {
      if (AllowedDev == DeviceID)
        return true;
    }
    return false;
  }

  /**
   * Validate function against mission profile constraints
   */
  bool validateFunction(Function &F, std::vector<std::string> &Violations) {
    bool Valid = true;

    // Check mission profile attribute match
    if (auto FuncProfile = getAttributeValue(F, "dsmil_mission_profile")) {
      if (*FuncProfile != ActiveProfile) {
        Violations.push_back("Function '" + F.getName().str() +
                           "' has dsmil_mission_profile(\"" + *FuncProfile +
                           "\") but compiling with -fdsmil-mission-profile=" +
                           ActiveProfile);
        Valid = false;
      }
    }

    // Check stage compatibility
    if (auto Stage = getAttributeValue(F, "dsmil_stage")) {
      if (!isStageAllowed(*Stage)) {
        Violations.push_back("Function '" + F.getName().str() +
                           "' uses stage '" + *Stage +
                           "' which is not allowed by mission profile '" +
                           ActiveProfile + "'");
        Valid = false;
      }
    }

    // Check layer policy
    if (auto Layer = getIntAttributeValue(F, "dsmil_layer")) {
      std::optional<std::string> RequiredROE;
      if (!isLayerAllowed(*Layer, RequiredROE)) {
        Violations.push_back("Function '" + F.getName().str() +
                           "' assigned to layer " + std::to_string(*Layer) +
                           " which is not allowed by mission profile '" +
                           ActiveProfile + "'");
        Valid = false;
      } else if (RequiredROE.has_value()) {
        // Check if function has required ROE
        auto FuncROE = getAttributeValue(F, "dsmil_roe");
        if (!FuncROE || *FuncROE != *RequiredROE) {
          Violations.push_back("Function '" + F.getName().str() +
                             "' on layer " + std::to_string(*Layer) +
                             " requires dsmil_roe(\"" + *RequiredROE +
                             "\") for mission profile '" + ActiveProfile + "'");
          Valid = false;
        }
      }
    }

    // Check device whitelist
    if (auto Device = getIntAttributeValue(F, "dsmil_device")) {
      if (!isDeviceAllowed(*Device)) {
        Violations.push_back("Function '" + F.getName().str() +
                           "' assigned to device " + std::to_string(*Device) +
                           " which is not whitelisted by mission profile '" +
                           ActiveProfile + "'");
        Valid = false;
      }
    }

    // Check quantum export restrictions
    if (!CurrentConfig.quantum_export) {
      if (F.hasFnAttribute("dsmil_quantum_candidate")) {
        Violations.push_back("Function '" + F.getName().str() +
                           "' marked as dsmil_quantum_candidate but mission profile '" +
                           ActiveProfile + "' forbids quantum_export");
        Valid = false;
      }
    }

    return Valid;
  }

public:
  DsmilMissionPolicyPass()
    : ActiveProfile(::MissionProfile.getValue()),
      ConfigPath(::MissionProfileConfig.getValue()),
      EnforcementMode(::MissionPolicyMode.getValue()) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    // If no mission profile specified, skip enforcement
    if (ActiveProfile.empty()) {
      LLVM_DEBUG(dbgs() << "[DSMIL Mission Policy] No mission profile specified, skipping\n");
      return PreservedAnalyses::all();
    }

    // If enforcement disabled, skip
    if (EnforcementMode == "disabled") {
      LLVM_DEBUG(dbgs() << "[DSMIL Mission Policy] Enforcement disabled\n");
      return PreservedAnalyses::all();
    }

    // Load mission profile configuration
    if (!loadMissionProfile(ActiveProfile)) {
      if (EnforcementMode == "enforce") {
        errs() << "[DSMIL Mission Policy] FATAL: Failed to load mission profile\n";
        report_fatal_error("Mission profile configuration error");
      }
      return PreservedAnalyses::all();
    }

    outs() << "[DSMIL Mission Policy] Enforcing mission profile: "
           << ActiveProfile << " (" << CurrentConfig.display_name << ")\n";
    outs() << "  Classification: " << CurrentConfig.classification << "\n";
    outs() << "  Operational Context: " << CurrentConfig.operational_context << "\n";
    outs() << "  Pipeline: " << CurrentConfig.pipeline << "\n";
    outs() << "  CT Enforcement: " << CurrentConfig.ct_enforcement << "\n";
    outs() << "  Telemetry Level: " << CurrentConfig.telemetry_level << "\n";

    // Validate all functions in module
    std::vector<std::string> AllViolations;
    int ViolationCount = 0;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      std::vector<std::string> FuncViolations;
      if (!validateFunction(F, FuncViolations)) {
        ViolationCount++;
        AllViolations.insert(AllViolations.end(),
                           FuncViolations.begin(),
                           FuncViolations.end());
      }
    }

    // Report violations
    if (!AllViolations.empty()) {
      errs() << "\n[DSMIL Mission Policy] Mission Profile Violations ("
             << ViolationCount << " functions affected):\n";
      for (const auto &V : AllViolations) {
        errs() << "  ERROR: " << V << "\n";
      }
      errs() << "\n";

      if (EnforcementMode == "enforce") {
        errs() << "[DSMIL Mission Policy] FATAL: Mission profile violations detected\n";
        errs() << "Hint: Check mission-profiles.json or adjust source annotations\n";
        report_fatal_error("Mission profile policy violations");
      } else {
        errs() << "[DSMIL Mission Policy] WARNING: Violations detected but enforcement mode is 'warn'\n";
      }
    } else {
      outs() << "[DSMIL Mission Policy] ✓ All functions comply with mission profile\n";
    }

    // Add module-level mission profile metadata
    LLVMContext &Ctx = M.getContext();
    M.setModuleFlag(Module::Error, "dsmil.mission_profile",
                   MDString::get(Ctx, ActiveProfile));
    M.setModuleFlag(Module::Error, "dsmil.mission_classification",
                   MDString::get(Ctx, CurrentConfig.classification));
    M.setModuleFlag(Module::Error, "dsmil.mission_pipeline",
                   MDString::get(Ctx, CurrentConfig.pipeline));

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // anonymous namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilMissionPolicyPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-mission-policy") {
            MPM.addPass(DsmilMissionPolicyPass());
            return true;
          }
          return false;
        });
    }
  };
}
