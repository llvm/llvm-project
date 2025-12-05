/**
 * @file DsmilTelecomPass.cpp
 * @brief DSLLVM Telecom/SS7/SIGTRAN Annotation Discovery & Manifest Pass
 *
 * Discovers telecom-related annotations (SS7/SIGTRAN roles, environments, etc.)
 * and generates compile-time manifests for Layer 8/9 awareness.
 *
 * Features:
 * - Telecom annotation discovery
 * - Telecom manifest JSON generation
 * - Integration with mission profiles
 * - Security policy enforcement (prod vs honeypot)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <fstream>

#define DEBUG_TYPE "dsmil-telecom"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableTelecomFlags(
    "dsmil-telecom-flags",
    cl::desc("Enable telecom annotation discovery and manifest generation"),
    cl::init(false));

static cl::opt<std::string> TelecomManifestPath(
    "dsmil-telecom-manifest-path",
    cl::desc("Path for telecom manifest JSON output"),
    cl::init(""));

static cl::opt<std::string> MissionProfile(
    "dsmil-mission-profile",
    cl::desc("Mission profile name"),
    cl::init(""));

namespace {

/**
 * Telecom metadata for a function
 */
struct TelecomFunctionMetadata {
    std::string name;
    uint8_t layer = 0;
    uint8_t device = 0;
    std::string stage;
    std::string telecom_stack;
    std::string ss7_role;
    std::string sigtran_role;
    std::string telecom_env;
    std::string sig_security;
    std::string telecom_if;
    std::string telecom_ep;
    std::string file;
    uint32_t line = 0;
};

/**
 * Module-level telecom metadata
 */
struct TelecomModuleMetadata {
    std::string module_id;
    std::string build_id;
    std::string provenance_id;
    std::string mission_profile;
    std::set<std::string> stacks;
    std::string default_env;
    std::string default_sig_security;
    std::vector<TelecomFunctionMetadata> functions;
};

/**
 * Telecom Annotation Discovery Pass
 */
class DsmilTelecomPass : public PassInfoMixin<DsmilTelecomPass> {
private:
    Module *M;
    std::string MissionProfileName;
    TelecomModuleMetadata ModuleMD;

    /**
     * Extract annotation value from function metadata
     */
    std::string extractAnnotation(Function &F, StringRef AttrName) {
        // Check for annotate metadata
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

        // Check instructions for annotations (Clang may attach to first instruction)
        if (!F.isDeclaration()) {
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (MDNode *MD = I.getMetadata("llvm.ptr.annotation")) {
                        for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                            if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                                StringRef Value = Str->getString();
                                if (Value.startswith(AttrName + "=")) {
                                    return Value.substr(AttrName.size() + 1).str();
                                }
                            }
                        }
                    }
                    break;  // Only check first instruction
                }
                break;
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
     * Extract DSMIL stage from function
     */
    std::string extractStage(Function &F) {
        if (F.hasFnAttribute("dsmil_stage")) {
            Attribute Attr = F.getFnAttribute("dsmil_stage");
            if (Attr.isStringAttribute()) {
                return Attr.getValueAsString().str();
            }
        }
        return "";
    }

    /**
     * Get source file and line from debug info
     */
    void getDebugLocation(Function &F, std::string &File, uint32_t &Line) {
        File = "unknown";
        Line = 0;

        if (!F.isDeclaration()) {
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (DILocation *Loc = I.getDebugLoc()) {
                        File = Loc->getFilename().str();
                        Line = Loc->getLine();
                        return;
                    }
                }
            }
        }
    }

    /**
     * Check if mission profile indicates telecom usage
     */
    bool isTelecomProfile(const std::string &Profile) {
        return Profile.find("ss7") != std::string::npos ||
               Profile.find("telco") != std::string::npos ||
               Profile.find("sigtran") != std::string::npos ||
               Profile.find("telecom") != std::string::npos;
    }

    /**
     * Collect telecom metadata from functions
     */
    void collectTelecomMetadata(Module &Mod) {
        bool hasTelecomCode = false;

        for (Function &F : Mod) {
            if (F.isDeclaration()) continue;

            TelecomFunctionMetadata MD;
            MD.name = F.getName().str();

            // Extract telecom annotations
            MD.telecom_stack = extractAnnotation(F, "dsmil.telecom_stack");
            MD.ss7_role = extractAnnotation(F, "dsmil.ss7_role");
            MD.sigtran_role = extractAnnotation(F, "dsmil.sigtran_role");
            MD.telecom_env = extractAnnotation(F, "dsmil.telecom_env");
            MD.sig_security = extractAnnotation(F, "dsmil.sig_security");
            MD.telecom_if = extractAnnotation(F, "dsmil.telecom_if");
            MD.telecom_ep = extractAnnotation(F, "dsmil.telecom_ep");

            // Extract DSMIL metadata
            MD.layer = extractLayer(F);
            MD.device = extractDevice(F);
            MD.stage = extractStage(F);
            getDebugLocation(F, MD.file, MD.line);

            // If function has any telecom annotation, add it
            if (!MD.telecom_stack.empty() || !MD.ss7_role.empty() ||
                !MD.sigtran_role.empty() || !MD.telecom_env.empty() ||
                !MD.sig_security.empty() || !MD.telecom_if.empty() ||
                !MD.telecom_ep.empty()) {
                hasTelecomCode = true;
                ModuleMD.functions.push_back(MD);

                // Track stacks
                if (!MD.telecom_stack.empty()) {
                    ModuleMD.stacks.insert(MD.telecom_stack);
                }

                // Track default environment (first non-empty)
                if (ModuleMD.default_env.empty() && !MD.telecom_env.empty()) {
                    ModuleMD.default_env = MD.telecom_env;
                }

                // Track default security (first non-empty)
                if (ModuleMD.default_sig_security.empty() && !MD.sig_security.empty()) {
                    ModuleMD.default_sig_security = MD.sig_security;
                }
            }
        }

        // Auto-enable if telecom profile detected
        if (!hasTelecomCode && isTelecomProfile(MissionProfileName)) {
            LLVM_DEBUG(dbgs() << "[Telecom] Auto-enabled due to telecom mission profile\n");
        }
    }

    /**
     * Generate telecom manifest JSON
     */
    void generateManifest(const std::string &OutputPath) {
        std::string ManifestPath = OutputPath;
        if (ManifestPath.empty()) {
            std::string ModuleName = M->getName().str();
            if (ModuleName.empty()) {
                ModuleName = "unknown";
            }
            ManifestPath = ModuleName + ".dsmil.telecom.json";
        }

        std::ofstream Out(ManifestPath);
        if (!Out.is_open()) {
            errs() << "Warning: Could not open telecom manifest file: " << ManifestPath << "\n";
            return;
        }

        Out << "{\n";
        Out << "  \"module_id\": \"" << ModuleMD.module_id << "\",\n";
        Out << "  \"build_id\": \"" << ModuleMD.build_id << "\",\n";
        Out << "  \"provenance_id\": \"" << ModuleMD.provenance_id << "\",\n";
        Out << "  \"mission_profile\": \"" << ModuleMD.mission_profile << "\",\n";
        Out << "  \"telecom\": {\n";

        // Stacks array
        Out << "    \"stacks\": [";
        bool first = true;
        for (const std::string &stack : ModuleMD.stacks) {
            if (!first) Out << ", ";
            Out << "\"" << stack << "\"";
            first = false;
        }
        Out << "],\n";

        Out << "    \"default_env\": \"" << ModuleMD.default_env << "\",\n";
        Out << "    \"default_sig_security\": \"" << ModuleMD.default_sig_security << "\"\n";
        Out << "  },\n";

        // Functions array
        Out << "  \"functions\": [\n";
        for (size_t i = 0; i < ModuleMD.functions.size(); i++) {
            const TelecomFunctionMetadata &MD = ModuleMD.functions[i];
            Out << "    {\n";
            Out << "      \"name\": \"" << MD.name << "\",\n";
            Out << "      \"layer\": " << (int)MD.layer << ",\n";
            Out << "      \"device\": " << (int)MD.device << ",\n";
            Out << "      \"stage\": \"" << MD.stage << "\"";

            if (!MD.telecom_stack.empty()) {
                Out << ",\n      \"telecom_stack\": \"" << MD.telecom_stack << "\"";
            }
            if (!MD.ss7_role.empty()) {
                Out << ",\n      \"ss7_role\": \"" << MD.ss7_role << "\"";
            }
            if (!MD.sigtran_role.empty()) {
                Out << ",\n      \"sigtran_role\": \"" << MD.sigtran_role << "\"";
            }
            if (!MD.telecom_env.empty()) {
                Out << ",\n      \"telecom_env\": \"" << MD.telecom_env << "\"";
            }
            if (!MD.sig_security.empty()) {
                Out << ",\n      \"sig_security\": \"" << MD.sig_security << "\"";
            }
            if (!MD.telecom_if.empty()) {
                Out << ",\n      \"telecom_if\": \"" << MD.telecom_if << "\"";
            }
            if (!MD.telecom_ep.empty()) {
                Out << ",\n      \"telecom_ep\": \"" << MD.telecom_ep << "\"";
            }

            Out << "\n    }";
            if (i < ModuleMD.functions.size() - 1) Out << ",";
            Out << "\n";
        }
        Out << "  ]\n";
        Out << "}\n";
        Out.close();

        outs() << "[DSMIL Telecom] Generated manifest: " << ManifestPath << "\n";
        outs() << "  Functions with telecom annotations: " << ModuleMD.functions.size() << "\n";
        outs() << "  Telecom stacks: " << ModuleMD.stacks.size() << "\n";
    }

    /**
     * Validate security policy (prod vs honeypot)
     */
    void validateSecurityPolicy() {
        bool hasProd = false;
        bool hasHoneypot = false;

        for (const auto &MD : ModuleMD.functions) {
            if (MD.telecom_env == "prod") {
                hasProd = true;
            } else if (MD.telecom_env == "honeypot") {
                hasHoneypot = true;
            }
        }

        if (hasProd && hasHoneypot) {
            errs() << "Warning: Module contains both production and honeypot code!\n";
            errs() << "  This may indicate a security policy violation.\n";
        }

        // Check mission profile consistency
        if (MissionProfileName.find("honeypot") != std::string::npos && hasProd) {
            errs() << "Error: Honeypot mission profile but production code detected!\n";
        }
    }

public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        // Check if enabled
        if (!EnableTelecomFlags && !isTelecomProfile(MissionProfile)) {
            return PreservedAnalyses::all();
        }

        M = &Mod;
        MissionProfileName = MissionProfile.empty() ? "default" : MissionProfile;

        // Initialize module metadata
        ModuleMD.module_id = Mod.getName().str();
        if (ModuleMD.module_id.empty()) {
            ModuleMD.module_id = "unknown";
        }
        ModuleMD.build_id = "0";  // Would extract from provenance
        ModuleMD.provenance_id = "0";  // Would extract from CNSA2
        ModuleMD.mission_profile = MissionProfileName;

        outs() << "[DSMIL Telecom] Analyzing module: " << Mod.getName() << "\n";

        // Collect telecom metadata
        collectTelecomMetadata(Mod);

        if (ModuleMD.functions.empty()) {
            LLVM_DEBUG(dbgs() << "[Telecom] No telecom annotations found\n");
            return PreservedAnalyses::all();
        }

        // Validate security policy
        validateSecurityPolicy();

        // Generate manifest
        generateManifest(TelecomManifestPath);

        return PreservedAnalyses::all();  // We don't modify IR, only generate manifest
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DsmilTelecomPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsmil-telecom") {
                        MPM.addPass(DsmilTelecomPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
