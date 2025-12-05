/**
 * @file DsmilTelemetryPass.cpp
 * @brief DSLLVM OT Telemetry Instrumentation Pass
 *
 * Instruments OT-critical functions and safety signals with telemetry calls
 * for high-value safety + OT visibility with minimal runtime overhead.
 *
 * Features:
 * - Function entry/exit instrumentation for OT-critical functions
 * - SES gate intent logging
 * - Safety signal update logging
 * - Telemetry manifest JSON generation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#define DEBUG_TYPE "dsmil-telemetry"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableOTTelemetry(
    "dsmil-ot-telemetry",
    cl::desc("Enable OT telemetry instrumentation"),
    cl::init(false));

static cl::opt<std::string> TelemetryManifestPath(
    "dsmil-telemetry-manifest-path",
    cl::desc("Path for telemetry manifest JSON output"),
    cl::init(""));

static cl::opt<std::string> MissionProfile(
    "dsmil-mission-profile",
    cl::desc("Mission profile name"),
    cl::init(""));

namespace {

/**
 * Function metadata extracted from annotations
 */
struct FunctionMetadata {
    std::string name;
    bool ot_critical = false;
    bool ses_gate = false;
    uint8_t authority_tier = 3;  // Default: analytics/advisory
    uint8_t layer = 0;
    uint8_t device = 0;
    std::string stage;
    std::string file;
    uint32_t line = 0;
};

/**
 * Safety signal metadata
 */
struct SafetySignalMetadata {
    std::string name;
    std::string type;
    uint8_t layer = 0;
    uint8_t device = 0;
    GlobalVariable *global = nullptr;
};

/**
 * OT Telemetry Instrumentation Pass
 */
class DsmilTelemetryPass : public PassInfoMixin<DsmilTelemetryPass> {
private:
    Module *M;
    std::string MissionProfileName;
    std::map<Function*, FunctionMetadata> FunctionMetadataMap;
    std::map<GlobalVariable*, SafetySignalMetadata> SafetySignals;
    std::vector<FunctionMetadata> ManifestFunctions;
    std::vector<SafetySignalMetadata> ManifestSignals;

    /**
     * Extract annotation value from attribute
     */
    std::string extractAnnotationValue(Function &F, StringRef AttrName) {
        if (F.hasFnAttribute(AttrName)) {
            Attribute Attr = F.getFnAttribute(AttrName);
            if (Attr.isStringAttribute()) {
                return Attr.getValueAsString().str();
            }
        }
        return "";
    }

    /**
     * Check if function has annotation attribute (via metadata)
     */
    bool hasAnnotation(Function &F, StringRef AttrName) {
        // Check for annotate metadata (Clang emits annotate attributes as metadata)
        if (MDNode *MD = F.getMetadata("llvm.ptr.annotation")) {
            for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                    if (Str->getString().startswith(AttrName)) {
                        return true;
                    }
                }
            }
        }
        
        // Also check function attributes (some compilers may use this)
        if (F.hasFnAttribute("annotate")) {
            Attribute Attr = F.getFnAttribute("annotate");
            if (Attr.isStringAttribute()) {
                StringRef Value = Attr.getValueAsString();
                return Value.startswith(AttrName);
            }
        }
        
        // Check for annotation in all instructions (Clang may attach to first instruction)
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
     * Collect function metadata
     */
    void collectFunctionMetadata(Function &F) {
        if (F.isDeclaration()) return;

        FunctionMetadata MD;
        MD.name = F.getName().str();

        // Check for OT annotations
        MD.ot_critical = hasAnnotation(F, "dsmil.ot_critical");
        MD.ses_gate = hasAnnotation(F, "dsmil.ses_gate");

        // Extract authority tier
        std::string tier_str = extractAnnotationParam(F, "dsmil.ot_tier");
        if (!tier_str.empty()) {
            MD.authority_tier = (uint8_t)std::stoi(tier_str);
        } else {
            // Default based on layer
            MD.layer = extractLayer(F);
            if (MD.layer <= 1) MD.authority_tier = 0;  // Safety kernel
            else if (MD.layer <= 3) MD.authority_tier = 1;  // High-impact control
            else if (MD.layer <= 6) MD.authority_tier = 2;  // Optimization
            else MD.authority_tier = 3;  // Analytics
        }

        MD.layer = extractLayer(F);
        MD.device = extractDevice(F);
        MD.stage = extractStage(F);
        getDebugLocation(F, MD.file, MD.line);

        if (MD.ot_critical || MD.ses_gate) {
            FunctionMetadataMap[&F] = MD;
            ManifestFunctions.push_back(MD);
        }
    }

    /**
     * Collect safety signal metadata
     */
    void collectSafetySignals(Module &Mod) {
        for (GlobalVariable &GV : Mod.globals()) {
            // Check for annotation metadata
            if (MDNode *MD = GV.getMetadata("llvm.ptr.annotation")) {
                for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                    if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                        StringRef Value = Str->getString();
                        if (Value.startswith("dsmil.safety_signal=")) {
                            SafetySignalMetadata SS;
                            SS.name = Value.substr(strlen("dsmil.safety_signal=")).str();
                            SS.global = &GV;
                            
                            // Determine type
                            Type *Ty = GV.getValueType();
                            if (Ty->isDoubleTy()) {
                                SS.type = "double";
                            } else if (Ty->isFloatTy()) {
                                SS.type = "float";
                            } else if (Ty->isIntegerTy()) {
                                SS.type = "int";
                            } else {
                                SS.type = "unknown";
                            }
                            
                            // Try to extract layer/device from module metadata or defaults
                            SS.layer = 0;
                            SS.device = 0;
                            SafetySignals[&GV] = SS;
                            ManifestSignals.push_back(SS);
                            break;
                        }
                    }
                }
            }
            
            // Also check attributes (fallback)
            if (GV.hasAttribute("annotate")) {
                Attribute Attr = GV.getAttribute("annotate");
                if (Attr.isStringAttribute()) {
                    StringRef Value = Attr.getValueAsString();
                    if (Value.startswith("dsmil.safety_signal=")) {
                        SafetySignalMetadata SS;
                        SS.name = Value.substr(strlen("dsmil.safety_signal=")).str();
                        SS.global = &GV;
                        
                        Type *Ty = GV.getValueType();
                        if (Ty->isDoubleTy()) {
                            SS.type = "double";
                        } else if (Ty->isFloatTy()) {
                            SS.type = "float";
                        } else if (Ty->isIntegerTy()) {
                            SS.type = "int";
                        } else {
                            SS.type = "unknown";
                        }
                        
                        SS.layer = 0;
                        SS.device = 0;
                        SafetySignals[&GV] = SS;
                        ManifestSignals.push_back(SS);
                    }
                }
            }
        }
    }

    /**
     * Get or create telemetry event function declaration
     */
    Function* getTelemetryEventFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {PointerType::getInt8PtrTy(M->getContext())},  // dsmil_telemetry_event_t*
            false);

        Function *F = M->getFunction("dsmil_telemetry_event");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage, "dsmil_telemetry_event", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Get or create safety signal update function declaration
     */
    Function* getSafetySignalUpdateFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {PointerType::getInt8PtrTy(M->getContext())},  // dsmil_telemetry_event_t*
            false);

        Function *F = M->getFunction("dsmil_telemetry_safety_signal_update");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage,
                               "dsmil_telemetry_safety_signal_update", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Create constant string in module
     */
    Constant* createStringConstant(const std::string &Str) {
        return ConstantDataArray::getString(M->getContext(), Str);
    }

    /**
     * Create telemetry event structure and call logging function
     */
    void createTelemetryCall(IRBuilder<> &Builder, FunctionMetadata &MD,
                            uint32_t EventType,
                            const std::string &ModuleID) {
        LLVMContext &Ctx = M->getContext();
        Function *TelemetryFn = getTelemetryEventFunction();

        // Create struct type matching dsmil_telemetry_event_t
        // Simplified: we'll create a minimal struct and initialize key fields
        // The runtime can extract additional info from debug metadata
        
        // Create string constants
        Constant *ModuleIDStr = ConstantDataArray::getString(Ctx, ModuleID, true);
        Constant *FuncIDStr = ConstantDataArray::getString(Ctx, MD.name, true);
        Constant *FileStr = ConstantDataArray::getString(Ctx, MD.file, true);
        Constant *StageStr = ConstantDataArray::getString(Ctx, MD.stage.empty() ? "" : MD.stage, true);
        Constant *ProfileStr = ConstantDataArray::getString(Ctx, MissionProfileName, true);

        // Create global variables for strings
        GlobalVariable *ModuleIDGV = new GlobalVariable(
            *M, ModuleIDStr->getType(), true, GlobalValue::PrivateLinkage,
            ModuleIDStr, "telemetry_module_id");
        GlobalVariable *FuncIDGV = new GlobalVariable(
            *M, FuncIDStr->getType(), true, GlobalValue::PrivateLinkage,
            FuncIDStr, "telemetry_func_id");
        GlobalVariable *FileGV = new GlobalVariable(
            *M, FileStr->getType(), true, GlobalValue::PrivateLinkage,
            FileStr, "telemetry_file");
        GlobalVariable *StageGV = new GlobalVariable(
            *M, StageStr->getType(), true, GlobalValue::PrivateLinkage,
            StageStr, "telemetry_stage");
        GlobalVariable *ProfileGV = new GlobalVariable(
            *M, ProfileStr->getType(), true, GlobalValue::PrivateLinkage,
            ProfileStr, "telemetry_profile");

        // Get pointers to string data
        Value *ModuleIDPtr = Builder.CreateConstGEP2_32(
            ModuleIDStr->getType(), ModuleIDGV, 0, 0);
        Value *FuncIDPtr = Builder.CreateConstGEP2_32(
            FuncIDStr->getType(), FuncIDGV, 0, 0);
        Value *FilePtr = Builder.CreateConstGEP2_32(
            FileStr->getType(), FileGV, 0, 0);
        Value *StagePtr = Builder.CreateConstGEP2_32(
            StageStr->getType(), StageGV, 0, 0);
        Value *ProfilePtr = Builder.CreateConstGEP2_32(
            ProfileStr->getType(), ProfileGV, 0, 0);

        // Create event struct type (simplified - using opaque pointer)
        // In a full implementation, we'd create the exact struct type
        // For now, pass a pointer that the runtime can interpret
        // The runtime will use debug info to fill in missing fields
        
        // Create a minimal event structure on the stack
        // We'll pass a pointer to a struct containing the essential fields
        StructType *EventTy = StructType::create(Ctx, "dsmil_telemetry_event");
        std::vector<Type*> Fields = {
            Type::getInt32Ty(Ctx),           // event_type
            PointerType::getInt8PtrTy(Ctx),  // module_id
            PointerType::getInt8PtrTy(Ctx),  // func_id
            PointerType::getInt8PtrTy(Ctx),  // file
            Type::getInt32Ty(Ctx),           // line
            Type::getInt8Ty(Ctx),            // layer
            Type::getInt8Ty(Ctx),            // device
            PointerType::getInt8PtrTy(Ctx),  // stage
            PointerType::getInt8PtrTy(Ctx),  // mission_profile
            Type::getInt8Ty(Ctx),            // authority_tier
            Type::getInt64Ty(Ctx),           // build_id
            Type::getInt64Ty(Ctx),           // provenance_id
            PointerType::getInt8PtrTy(Ctx),  // signal_name
            Type::getDoubleTy(Ctx),         // signal_value
            Type::getDoubleTy(Ctx),         // signal_min
            Type::getDoubleTy(Ctx)          // signal_max
        };
        EventTy->setBody(Fields);

        // Allocate event structure
        AllocaInst *EventAlloca = Builder.CreateAlloca(EventTy, nullptr, "telemetry_event");

        // Initialize event structure fields
        Value *Zero32 = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
        Value *Zero8 = ConstantInt::get(Type::getInt8Ty(Ctx), 0);
        Value *Zero64 = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
        Value *EventTypeVal = ConstantInt::get(Type::getInt32Ty(Ctx), EventType);
        Value *LineVal = ConstantInt::get(Type::getInt32Ty(Ctx), MD.line);
        Value *LayerVal = ConstantInt::get(Type::getInt8Ty(Ctx), MD.layer);
        Value *DeviceVal = ConstantInt::get(Type::getInt8Ty(Ctx), MD.device);
        Value *TierVal = ConstantInt::get(Type::getInt8Ty(Ctx), MD.authority_tier);
        Value *NullPtr = ConstantPointerNull::get(PointerType::getInt8PtrTy(Ctx));
        Value *ZeroDouble = ConstantFP::get(Type::getDoubleTy(Ctx), 0.0);

        // Store fields
        Builder.CreateStore(EventTypeVal, Builder.CreateStructGEP(EventTy, EventAlloca, 0));
        Builder.CreateStore(ModuleIDPtr, Builder.CreateStructGEP(EventTy, EventAlloca, 1));
        Builder.CreateStore(FuncIDPtr, Builder.CreateStructGEP(EventTy, EventAlloca, 2));
        Builder.CreateStore(FilePtr, Builder.CreateStructGEP(EventTy, EventAlloca, 3));
        Builder.CreateStore(LineVal, Builder.CreateStructGEP(EventTy, EventAlloca, 4));
        Builder.CreateStore(LayerVal, Builder.CreateStructGEP(EventTy, EventAlloca, 5));
        Builder.CreateStore(DeviceVal, Builder.CreateStructGEP(EventTy, EventAlloca, 6));
        Builder.CreateStore(StagePtr, Builder.CreateStructGEP(EventTy, EventAlloca, 7));
        Builder.CreateStore(ProfilePtr, Builder.CreateStructGEP(EventTy, EventAlloca, 8));
        Builder.CreateStore(TierVal, Builder.CreateStructGEP(EventTy, EventAlloca, 9));
        Builder.CreateStore(Zero64, Builder.CreateStructGEP(EventTy, EventAlloca, 10));  // build_id
        Builder.CreateStore(Zero64, Builder.CreateStructGEP(EventTy, EventAlloca, 11));  // provenance_id
        Builder.CreateStore(NullPtr, Builder.CreateStructGEP(EventTy, EventAlloca, 12));  // signal_name
        Builder.CreateStore(ZeroDouble, Builder.CreateStructGEP(EventTy, EventAlloca, 13));  // signal_value
        Builder.CreateStore(ZeroDouble, Builder.CreateStructGEP(EventTy, EventAlloca, 14));  // signal_min
        Builder.CreateStore(ZeroDouble, Builder.CreateStructGEP(EventTy, EventAlloca, 15));  // signal_max

        // Cast to void* for function call
        Value *EventPtr = Builder.CreateBitCast(EventAlloca, PointerType::getInt8PtrTy(Ctx));
        Builder.CreateCall(TelemetryFn, {EventPtr});
    }

    /**
     * Instrument function entry
     */
    void instrumentFunctionEntry(Function &F, FunctionMetadata &MD) {
        if (F.isDeclaration()) return;

        BasicBlock &EntryBB = F.getEntryBlock();
        IRBuilder<> Builder(&EntryBB, EntryBB.begin());

        std::string ModuleID = M->getName().str();
        if (ModuleID.empty()) {
            ModuleID = "unknown_module";
        }

        uint32_t EventType = 1;  // DSMIL_TELEMETRY_OT_PATH_ENTRY
        if (MD.ses_gate) {
            EventType = 3;  // DSMIL_TELEMETRY_SES_INTENT
        }

        createTelemetryCall(Builder, MD, EventType, ModuleID);
    }

    /**
     * Instrument function exits
     */
    void instrumentFunctionExits(Function &F, FunctionMetadata &MD) {
        if (F.isDeclaration()) return;

        std::string ModuleID = M->getName().str();
        if (ModuleID.empty()) {
            ModuleID = "unknown_module";
        }

        for (BasicBlock &BB : F) {
            Instruction *Term = BB.getTerminator();
            if (isa<ReturnInst>(Term) || isa<ResumeInst>(Term)) {
                IRBuilder<> Builder(Term);
                createTelemetryCall(Builder, MD, 2, ModuleID);  // DSMIL_TELEMETRY_OT_PATH_EXIT
            }
        }
    }

    /**
     * Instrument safety signal stores
     */
    void instrumentSafetySignals(Function &F) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
                    Value *Ptr = SI->getPointerOperand();
                    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
                        if (SafetySignals.count(GV)) {
                            SafetySignalMetadata &SS = SafetySignals[GV];
                            // Instrument after store
                            IRBuilder<> Builder(&I);
                            Builder.SetInsertPoint(&BB, ++BasicBlock::iterator(&I));

                            // Create telemetry call for safety signal update
                            Function *SignalFn = getSafetySignalUpdateFunction();
                            Value *NullPtr = ConstantPointerNull::get(PointerType::getInt8PtrTy(M->getContext()));
                            Builder.CreateCall(SignalFn, {NullPtr});
                        }
                    }
                }
            }
        }
    }

    /**
     * Generate telemetry manifest JSON
     */
    void generateManifest(const std::string &OutputPath) {
        std::string ManifestPath = OutputPath;
        if (ManifestPath.empty()) {
            std::string ModuleName = M->getName().str();
            if (ModuleName.empty()) {
                ModuleName = "unknown";
            }
            ManifestPath = ModuleName + ".dsmil.telemetry.json";
        }

        std::ofstream Out(ManifestPath);
        if (!Out.is_open()) {
            errs() << "Warning: Could not open manifest file: " << ManifestPath << "\n";
            return;
        }

        Out << "{\n";
        Out << "  \"module_id\": \"" << M->getName().str() << "\",\n";
        Out << "  \"build_id\": \"0\",\n";  // Would extract from provenance
        Out << "  \"provenance_id\": \"0\",\n";  // Would extract from CNSA2
        Out << "  \"mission_profile\": \"" << MissionProfileName << "\",\n";
        Out << "  \"functions\": [\n";

        for (size_t i = 0; i < ManifestFunctions.size(); i++) {
            const FunctionMetadata &MD = ManifestFunctions[i];
            Out << "    {\n";
            Out << "      \"name\": \"" << MD.name << "\",\n";
            Out << "      \"layer\": " << (int)MD.layer << ",\n";
            Out << "      \"device\": " << (int)MD.device << ",\n";
            Out << "      \"stage\": \"" << MD.stage << "\",\n";
            Out << "      \"ot_critical\": " << (MD.ot_critical ? "true" : "false") << ",\n";
            Out << "      \"authority_tier\": " << (int)MD.authority_tier << ",\n";
            Out << "      \"ses_gate\": " << (MD.ses_gate ? "true" : "false") << "\n";
            Out << "    }";
            if (i < ManifestFunctions.size() - 1) Out << ",";
            Out << "\n";
        }

        Out << "  ],\n";
        Out << "  \"safety_signals\": [\n";

        for (size_t i = 0; i < ManifestSignals.size(); i++) {
            const SafetySignalMetadata &SS = ManifestSignals[i];
            Out << "    {\n";
            Out << "      \"name\": \"" << SS.name << "\",\n";
            Out << "      \"type\": \"" << SS.type << "\",\n";
            Out << "      \"layer\": " << (int)SS.layer << ",\n";
            Out << "      \"device\": " << (int)SS.device << "\n";
            Out << "    }";
            if (i < ManifestSignals.size() - 1) Out << ",";
            Out << "\n";
        }

        Out << "  ]\n";
        Out << "}\n";
        Out.close();

        outs() << "[DSMIL Telemetry] Generated manifest: " << ManifestPath << "\n";
    }

public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        if (!EnableOTTelemetry) {
            return PreservedAnalyses::all();
        }

        M = &Mod;
        MissionProfileName = MissionProfile.empty() ? "default" : MissionProfile;

        outs() << "[DSMIL OT Telemetry] Instrumenting module: " << Mod.getName() << "\n";

        // Collect metadata from all functions
        for (Function &F : Mod) {
            collectFunctionMetadata(F);
        }

        // Collect safety signals
        collectSafetySignals(Mod);

        outs() << "  OT-Critical Functions: " << FunctionMetadataMap.size() << "\n";
        outs() << "  Safety Signals: " << SafetySignals.size() << "\n";

        // Instrument functions
        for (auto &Pair : FunctionMetadataMap) {
            Function *F = Pair.first;
            FunctionMetadata &MD = Pair.second;

            if (MD.ot_critical) {
                instrumentFunctionEntry(*F, MD);
                instrumentFunctionExits(*F, MD);
            }

            if (MD.ses_gate) {
                instrumentFunctionEntry(*F, MD);  // SES intent logging
            }

            // Instrument safety signals in this function
            instrumentSafetySignals(*F);
        }

        // Generate manifest
        generateManifest(TelemetryManifestPath);

        return PreservedAnalyses::none();  // We modified the IR
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DsmilTelemetryPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsmil-telemetry") {
                        MPM.addPass(DsmilTelemetryPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
