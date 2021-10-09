//===- ClangBuildSelectLink.cpp  ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  clang-build-select-link a.bc b.bc c.bc -o merged.bc
//
// This utility merges all the bc files, then build select_outline_wrapper
// which is a big switch statement that depends on hash values.
// Then it goes back and marks each external function as linkOnceODR
// so the optimnization pass will remove wrappers and external functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

#include <map>
#include <memory>
#include <utility>

using namespace llvm;

static cl::list<std::string> InputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input bitcode files>"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool> Verbose("v",
                             cl::desc("Print information about actions taken"),
                             cl::init(false));

static cl::opt<bool> DirectCalls("d", cl::desc("Enable direct calls"),
                                 cl::init(true));

static cl::opt<bool> BuiltinCode("mlink-builtin-bitcode", cl::desc("Ignore option"),
                                 cl::ZeroOrMore, cl::init(true));

static ExitOnError ExitOnErr;

static bool loadArFile(const char *argv0, const std::string ArchiveName,
                       LLVMContext &Context, Linker &L, unsigned OrigFlags,
                       unsigned ApplicableFlags) {
  if (Verbose)
    errs() << "Reading library archive file '" << ArchiveName
           << "' to memory\n";
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFile(ArchiveName, -1, false);
  if (std::error_code EC = Buf.getError()) {
    if (Verbose)
      errs() << "Skipping archive : File not found " << ArchiveName << "\n";
    return false;
  } else {
    Error Err = Error::success();
    object::Archive Archive(Buf.get()->getMemBufferRef(), Err);
    object::Archive *ArchivePtr = &Archive;
    EC = errorToErrorCode(std::move(Err));
    if (Err) {
      if (Verbose)
        errs() << "Skipping archive : Empty file found " << ArchiveName << "\n";
      return false;
    }
    for (auto &C : ArchivePtr->children(Err)) {
      Expected<StringRef> ename = C.getName();
      if (Error E = ename.takeError()) {
        errs() << argv0 << ": ";
        WithColor::error()
            << " could not get member name of archive library failed'"
            << ArchiveName << "'\n";
        return false;
      };
      std::string goodname = ename.get().str();
      if (Verbose)
        errs() << "Parsing member '" << goodname
               << "' of archive library to module.\n";
      SMDiagnostic ParseErr;
      Expected<MemoryBufferRef> MemBuf = C.getMemoryBufferRef();
      if (Error E = MemBuf.takeError()) {
        errs() << argv0 << ": ";
        WithColor::error() << " loading memory for member '" << goodname
                           << "' of archive library failed'" << ArchiveName
                           << "'\n";
        return false;
      };

      std::unique_ptr<Module> M = parseIR(MemBuf.get(), ParseErr, Context);
      if (!M.get()) {
        errs() << argv0 << ": ";
        WithColor::error() << " parsing member '" << goodname
                           << "' of archive library failed'" << ArchiveName
                           << "'\n";
        return false;
      }
      if (Verbose)
        errs() << "Linking member '" << goodname << "' of archive library.\n";
      if (M.get()->getTargetTriple() != "") {
        bool Err = L.linkInModule(std::move(M), ApplicableFlags);
        if (Err)
          return false;
      }
      ApplicableFlags = OrigFlags;
    } // end for each child
    if (Err) {
      if (Verbose)
        errs() << "Skipping archive : Linking Error " << ArchiveName << "\n";
      return false;
    }
  }
  return true;
}

// Read bitcode file and return Module.
static std::unique_ptr<Module>
loadBcFile(const char *argv0, const std::string &FN, LLVMContext &Context) {
  SMDiagnostic Err;
  if (Verbose)
    errs() << "Loading '" << FN << "'\n";
  std::unique_ptr<Module> Result;
  Result = parseIRFile(FN, Err, Context);

  if (!Result) {
    Err.print(argv0, errs());
    return nullptr;
  }

  ExitOnErr(Result->materializeMetadata());
  UpgradeDebugInfo(*Result);

  return Result;
}

static bool linkFiles(const char *argv0, LLVMContext &Context, Linker &L,
                      const cl::list<std::string> &Files, unsigned Flags) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;
  // Similar to some flags, internalization doesn't apply to the first file.
  for (const auto &File : Files) {
    const char *Ext = strrchr(File.c_str(), '.');
    if (!strncmp(Ext, ".a", 2)) {
      if (Verbose)
        errs() << "Loading library archive file'" << File << "'\n";
      bool loadArSuccess =
          loadArFile(argv0, File, Context, L, Flags, ApplicableFlags);
      if (!loadArSuccess)
        continue;
    } else {
      if (Verbose)
        errs() << "Loading bc file'" << File << "'\n";
      std::unique_ptr<Module> M = loadBcFile(argv0, File, Context);
      if (!M.get()) {
        errs() << argv0 << ": ";
        WithColor::error() << " loading file '" << File << "'\n";
        return false;
      }
      if (Verbose)
        errs() << "Linking bc File'" << File << "' to module.\n";
      if (M.get()->getTargetTriple() != "") {
        bool Err = L.linkInModule(std::move(M), ApplicableFlags);
        if (Err)
          return false;
      }
    }
    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }
  return true;
}

// Rewrite select_outline_wrapper calls, to be direct calls.
//   @_HASHW_DeclareSharedMemory_cpp__omp_outlined___wrapper =
//     local_unnamed_addr addrspace(4) constant i64 -4874776124079246075
//   call void @select_outline_wrapper(i16 0, i32 %6, i64 -4874776124079246075)
// becomes
//   call void @DeclareSharedMemory_cpp__omp_outlined___wrapper(i16 0, i32 %6)
//
// We still neeed to generate the switch statement for the wrapper as there are
// situations where the callee is not known at compile time.
// See smoke/target-in-other-source as an example.
static bool rewriteSelectCalls(Module *MOUT, LLVMContext &Ctx) {

  // Build list of global variables and hashes as keys
  std::map<uint64_t, Function *> HashGlobalsMap;
  for (Module::global_iterator globi = MOUT->global_begin(),
                               e = MOUT->global_end();
       globi != e; ++globi) {
    GlobalVariable *GV = &*globi;
    if (GV->hasName()) {
      StringRef name = GV->getName();
      if (name.startswith("_HASHW_")) {
        // Get the actual Function
        StringRef wrapperName = GV->getName().substr(7);
        llvm::Function *F = MOUT->getFunction(wrapperName);
        // Get the 64 bit hash code from the GV to define the hash index
        const APInt &value = GV->getInitializer()->getUniqueInteger();
        const uint64_t *rawvalue = value.getRawData();
        // Create a map from hash to name
        HashGlobalsMap.insert(
            std::pair<uint64_t, Function *>((long long)*rawvalue, F));
        if (Verbose)
          fprintf(stderr, "Added hash %lld for function %s\n",
                  (long long)*rawvalue, F->getName().str().c_str());
      }
    }
  }

  // Linear scan on all instructions looking for calls to select_outline_wrapper
  llvm::IRBuilder<> Builder(Ctx);
  for (auto &F : MOUT->functions()) {
    for (auto &BB : F) {
      SmallVector<Instruction *, 4> DelInstrs;
      for (auto &I : BB) {
        // Must be a call
        if (!isa<CallInst>(&I))
          continue;
        const CallInst *CI = dyn_cast<CallInst>(&I);
        Function *CF = CI->getCalledFunction();
        StringRef name = CF ? CF->getName() : "";
        if (!name.startswith("select_outline_wrapper"))
          continue;
        Value *V0 = CI->getArgOperand(0);
        Value *V1 = CI->getArgOperand(1);
        // Get the hash and pull out the Function
        Value *V2 = CI->getArgOperand(2);
        ConstantInt *VCI = dyn_cast<ConstantInt>(V2);
        if (!VCI || VCI->getBitWidth() > 64)
          continue;
        uint64_t Hash = VCI->getSExtValue();
        Function *NewF = HashGlobalsMap[Hash];
        assert(NewF && "Could not find NewF");
        // Create the new call
        if (Verbose)
          fprintf(stderr, "Rewriting select_outline_wrapper to %s\n",
                  NewF->getName().str().c_str());
        Builder.SetInsertPoint(&I);
        Builder.CreateCall(NewF, {V0, V1});
        DelInstrs.push_back(&I);
      }
      // Remove the old instructions.
      if (!DelInstrs.empty()) {
        for (auto &I : DelInstrs)
          I->eraseFromParent();
      }
    }
  }
  return true;
}

static bool buildSelectFunction(Module *MOUT, LLVMContext &Ctx) {

  // Find select_outline_wrapper decl, because we are about to define it.
  llvm::IRBuilder<> Builder(Ctx);
  llvm::Function *Fn = MOUT->getFunction("select_outline_wrapper");
  if (!Fn) {
    if (Verbose)
      errs() << "No calls to select_outline_wrapper, skipping generation\n";
    return true;
  }

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(Ctx, "entry", Fn, nullptr);
  llvm::BasicBlock *exitbb = llvm::BasicBlock::Create(Ctx, "exit", Fn, nullptr);
  Builder.SetInsertPoint(entry);
  llvm::BasicBlock *defaultbb =
      llvm::BasicBlock::Create(Ctx, "default", Fn, nullptr);
  Builder.SetInsertPoint(defaultbb);
  Builder.CreateBr(exitbb);
  SmallVector<llvm::Value *, 4> PArgs;
  for (auto &Arg : Fn->args())
    PArgs.push_back(&Arg);
  SmallVector<llvm::Value *, 4> CArgs = {PArgs[0], PArgs[1]};
  Builder.SetInsertPoint(entry);

  // Count and build list of global variables
  llvm::SmallVector<llvm::GlobalVariable *, 16> hashglobals;
  unsigned hash_count = 0;
  for (Module::global_iterator globi = MOUT->global_begin(),
                               e = MOUT->global_end();
       globi != e; ++globi) {
    GlobalVariable *GV = &*globi;
    if (GV->hasName()) {
      StringRef name = GV->getName();
      if (name.startswith("_HASHW_")) {
        hash_count++;
        hashglobals.push_back(GV);
      }
    }
  }

  // Create the switch statement
  llvm::SwitchInst *Switch =
      Builder.CreateSwitch(PArgs[2], defaultbb, hash_count);

  if (Verbose)
    errs() << "Generating function " << Fn->getName().str() << " with "
           << hash_count << " case(s). \n";

  // Build a switch case for each hashglobal to call the function
  for (llvm::GlobalVariable *GV : hashglobals) {
    StringRef wrapperName = GV->getName().substr(7);
    llvm::Function *F = MOUT->getFunction(wrapperName);
    if (!F) {
      llvm::errs() << "\n FATAL ERROR, could not find function:\n";
      llvm::errs() << wrapperName.str().c_str() << "\n";
      return false;
    }
    // Get the 64bit hash code from the GV to define the value for the case
    const APInt &value = GV->getInitializer()->getUniqueInteger();
    const uint64_t *rawvalue = value.getRawData();

    Builder.SetInsertPoint(entry);
    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(Ctx, "BB" + wrapperName, Fn, nullptr);
    Builder.SetInsertPoint(BB);

    // Create the call to the actual wrapper function for this case
    Builder.CreateCall(F, CArgs);
    Builder.CreateBr(exitbb);
    llvm::Value *val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx),
                                              (long long)*rawvalue);
    Switch->addCase(cast<llvm::ConstantInt>(val), BB);
  }

  // Finish and verify the select_outline_wrapper function
  Builder.SetInsertPoint(exitbb);
  llvm::ReturnInst::Create(Ctx, nullptr, exitbb);
  Fn->setCallingConv(CallingConv::C);
  Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
  if (llvm::verifyFunction(*Fn)) {
    llvm::errs() << "Error in verifying function:\n";
    llvm::errs() << Fn->getName().str().c_str() << "\n";
    return false;
  }

  if (Verbose) {
    errs() << "Generated function is \n";
#ifndef NDEBUG
    Fn->dump();
#endif
  }
  return true;
}

static bool convertExternsToLinkOnce(Module *MOUT, LLVMContext &Ctx) {
  // Convert all external functions to LinkOnceODR so they get inlined
  // and removed by the optimizer in the next HIP driver step.
  // After next opt step, only kernels will exist
  for (Module::iterator i = MOUT->begin(), e = MOUT->end(); i != e; ++i) {
    llvm::Function *F = &*i;
    if (!i->isDeclaration()) {
      if (Verbose)
        errs() << "Function attribute cleanup for\'"
               << F->getName().str().c_str() << "\' \n";
      if (i->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
        F->removeFnAttr(llvm::Attribute::OptimizeNone);
      } else {
        F->setLinkage(GlobalValue::LinkOnceODRLinkage);
        F->setVisibility(GlobalValue::ProtectedVisibility);
        F->removeFnAttr(llvm::Attribute::OptimizeNone);
        F->removeFnAttr(llvm::Attribute::NoInline);
        F->addFnAttr(llvm::Attribute::AlwaysInline);
      }
    }
  }
  return true;
}

static bool runInliner(Module *MOUT, LLVMContext &Ctx) {
  legacy::PassManager PM;
  PassManagerBuilder Builder;
  Builder.Inliner = createAlwaysInlinerLegacyPass();
  Builder.populateModulePassManager(PM);
  PM.run(*MOUT);
  return true;
}

static bool removeStackSaveRestore(Module *MOUT, LLVMContext &Ctx) {
  StringRef fName("llvm.stacksave");
  llvm::Function *F = MOUT->getFunction(fName);
  if (F) {
    printf("\n\n FOUND stacksave \n");
#ifndef NDEBUG
    F->dump();
#endif
  }
  return true;
}

int main(int argc, char **argv) {
  InitLLVM InitX(argc, argv);
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  LLVMContext Context;

  cl::ParseCommandLineOptions(argc, argv, "clang-build-select-link\n");

  auto Composite = std::make_unique<Module>("clang-build-select-link", Context);
  Linker L(*Composite);

  unsigned Flags = Linker::Flags::None;

  if (!linkFiles(argv[0], Context, L, InputFilenames, Flags))
    return 1;

  Module *MOUT = &*Composite;
#if 0
  if (DirectCalls) {
    if (!rewriteSelectCalls(MOUT, Context))
      return 1;
  }

  if (!buildSelectFunction(MOUT, Context))
    return 1;
#endif
  if (!convertExternsToLinkOnce(MOUT, Context))
    return 1;
#if 0
  if (!runInliner(MOUT, Context))
    return 1;
#endif

  if (!removeStackSaveRestore(MOUT, Context))
    return 1;

  std::error_code EC;
  ToolOutputFile Out(OutputFilename, EC, sys::fs::OF_None);
  if (EC) {
    WithColor::error() << EC.message() << '\n';
    return 1;
  }

  if (verifyModule(*Composite, &errs())) {
    errs() << argv[0] << ": ";
    WithColor::error() << "linked module is broken!\n";
    return 1;
  }

  if (Verbose)
    errs() << "Writing merged bitcode...\n";

  WriteBitcodeToFile(*Composite, Out.os(), false);

  Out.keep();

  return 0;
}
