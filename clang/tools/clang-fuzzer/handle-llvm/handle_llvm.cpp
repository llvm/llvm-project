//==-- handle_llvm.cpp - Helper function for Clang fuzzers -----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements HandleLLVM for use by the Clang fuzzers. Mimics the llc tool to
// compile an LLVM IR file to X86_64 assembly.
//
//===----------------------------------------------------------------------===//

#include "handle_llvm.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdlib>

using namespace llvm;

static void getOptLevel(const std::vector<const char *> &ExtraArgs,
                              CodeGenOpt::Level &OLvl) {
  // Find the optimization level from the command line args
  OLvl = CodeGenOpt::Default;
  for (auto &A : ExtraArgs) {
    if (A[0] == '-' && A[1] == 'O') {
      switch(A[2]) {
        case '0': OLvl = CodeGenOpt::None; break;
        case '1': OLvl = CodeGenOpt::Less; break;
        case '2': OLvl = CodeGenOpt::Default; break;
        case '3': OLvl = CodeGenOpt::Aggressive; break;
        default:
          errs() << "error: opt level must be between 0 and 3.\n";
          std::exit(1);
      }
    }
  }
}

void clang_fuzzer::HandleLLVM(const std::string &S,
                              const std::vector<const char *> &ExtraArgs) {
  // Parse ExtraArgs to set the optimization level
  CodeGenOpt::Level OLvl;
  getOptLevel(ExtraArgs, OLvl);

  // Set the Module to include the the IR code to be compiled
  SMDiagnostic Err;

  LLVMContext Context;
  std::unique_ptr<Module> M = parseIR(MemoryBufferRef(S, "IR"), Err, Context);
  if (!M) {
    errs() << "error: could not parse IR!\n";
    std::exit(1);
  }

  // Create a new Target
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(
      sys::getDefaultTargetTriple(), Error);
  if (!TheTarget) {
    errs() << Error;
    std::exit(1);
  }

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  // Create a new Machine
  std::string CPUStr = getCPUStr();
  std::string FeaturesStr = getFeaturesStr();
  std::unique_ptr<TargetMachine> Target(TheTarget->createTargetMachine(
      sys::getDefaultTargetTriple(), CPUStr, FeaturesStr, Options,
      getRelocModel(), getCodeModel(), OLvl));

  // Create a new PassManager
  legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  M->setDataLayout(Target->createDataLayout());
 
  // Make sure the Module has no errors
  if (verifyModule(*M, &errs())) {
    errs() << "error: input module is broken!\n";
    std::exit(1);
  } 

  setFunctionAttributes(CPUStr, FeaturesStr, *M);
  
  raw_null_ostream OS;
  Target->addPassesToEmitFile(PM, OS, nullptr, TargetMachine::CGFT_ObjectFile,
                              false);
  PM.run(*M);

  return;
}

