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

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

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
    if (!llvm::sys::fs::exists(File)) {
      errs() << "Warning: clang-build-select-link, file: '" << File <<
	     "'\n         Input file does not exist. File will be skipped.\n";
      continue;
    }
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

static bool convertExternsToLinkOnce(Module *MOUT, LLVMContext &Ctx) {
  for (Module::iterator i = MOUT->begin(), e = MOUT->end(); i != e; ++i) {
    llvm::Function *F = &*i;
    if (!i->isDeclaration()) {
      if (i->getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
        // defined function is not an AMD kernel
        if (Verbose)
          errs() << "Modifying Function attributes for function \'"
                 << F->getName().str().c_str() << "\' \n";
        // Convert functions to LinkOnceODR with protected visibility
        F->setLinkage(GlobalValue::LinkOnceODRLinkage);
        F->setVisibility(GlobalValue::ProtectedVisibility);
        if (!strncmp(F->getName().str().c_str(), "__ockl_devmem_request",
                     strlen("__ockl_devmem_request")))
          continue;
        if (!strncmp(F->getName().str().c_str(), "__ockl_dm_alloc",
                     strlen("__ockl_dm_alloc")))
          continue;
        if (!strncmp(F->getName().str().c_str(), "__ockl_dm_dealloc",
                     strlen("__ockl_dm_dealloc")))
          continue;
        if (!strncmp(F->getName().str().c_str(), "hostexec_invoke",
                     strlen("hostexec_invoke")))
          continue;
        // all other functions
        if (!F->hasOptNone()) {
          F->removeFnAttr(llvm::Attribute::OptimizeNone);
          F->removeFnAttr(llvm::Attribute::NoInline);
          F->addFnAttr(llvm::Attribute::AlwaysInline);
	}
      } else {
        // defined function is an AMD kernel
        if (F->getName().starts_with("__nv_")) {
          // Assume FORTRAN kernels start with __nv_
          if (Verbose)
            errs() << "Kernel attributes added to FORTRAN kernel\'"
                   << F->getName().str().c_str() << "\' \n";
          // Function Attrs: convergent mustprogress norecurse, nounwind
          F->addFnAttr(llvm::Attribute::Convergent);
          F->addFnAttr(llvm::Attribute::MustProgress);
          F->addFnAttr(llvm::Attribute::NoRecurse);
          F->addFnAttr(llvm::Attribute::NoUnwind);
          F->setVisibility(GlobalValue::ProtectedVisibility);
        }
      }
    }
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
  if (!convertExternsToLinkOnce(MOUT, Context))
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
