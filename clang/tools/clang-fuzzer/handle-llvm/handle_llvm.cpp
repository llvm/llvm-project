//==-- handle_llvm.cpp - Helper function for Clang fuzzers -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements HandleLLVM for use by the Clang fuzzers. First runs a loop
// vectorizer optimization pass over the given IR code. Then mimics lli on both
// versions to JIT the generated code and execute it. Currently, functions are 
// executed on dummy inputs.
//
//===----------------------------------------------------------------------===//

#include "handle_llvm.h"
#include "input_arrays.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;

// Define a type for the functions that are compiled and executed
typedef void (*LLVMFunc)(int*, int*, int*, int);

// Helper function to parse command line args and find the optimization level
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

static void ErrorAndExit(std::string message) {
  errs()<< "ERROR: " << message << "\n";
  std::exit(1);
}

// Helper function to add optimization passes to the TargetMachine at the 
// specified optimization level, OptLevel
static void AddOptimizationPasses(legacy::PassManagerBase &MPM,
                                  CodeGenOpt::Level OptLevel,
                                  unsigned SizeLevel) {
  // Create and initialize a PassManagerBuilder
  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;
  Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel, false);
  Builder.LoopVectorize = true;
  Builder.populateModulePassManager(MPM);
}

// Mimics the opt tool to run an optimization pass over the provided IR
static std::string OptLLVM(const std::string &IR, CodeGenOpt::Level OLvl) {
  // Create a module that will run the optimization passes
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseIR(MemoryBufferRef(IR, "IR"), Err, Context);
  if (!M || verifyModule(*M, &errs()))
    ErrorAndExit("Could not parse IR");

  Triple ModuleTriple(M->getTargetTriple());
  const TargetOptions Options =
      codegen::InitTargetOptionsFromCodeGenFlags(ModuleTriple);
  std::string E;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), ModuleTriple, E);
  if (!TheTarget)
    ErrorAndExit(E);

  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      M->getTargetTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(),
      Options, codegen::getExplicitRelocModel(),
      codegen::getExplicitCodeModel(), OLvl));
  if (!TM)
    ErrorAndExit("Could not create target machine");

  codegen::setFunctionAttributes(codegen::getCPUStr(),
                                 codegen::getFeaturesStr(), *M);

  legacy::PassManager Passes;
  
  Passes.add(new TargetLibraryInfoWrapperPass(ModuleTriple));
  Passes.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  LLVMTargetMachine &LTM = static_cast<LLVMTargetMachine &>(*TM);
  Passes.add(LTM.createPassConfig(Passes));

  Passes.add(createVerifierPass());

  AddOptimizationPasses(Passes, OLvl, 0);

  // Add a pass that writes the optimized IR to an output stream
  std::string outString;
  raw_string_ostream OS(outString);
  Passes.add(createPrintModulePass(OS, "", false));

  Passes.run(*M);

  return outString;
}

// Takes a function and runs it on a set of inputs
// First determines whether f is the optimized or unoptimized function
static void RunFuncOnInputs(LLVMFunc f, int Arr[kNumArrays][kArraySize]) {
  for (int i = 0; i < kNumArrays / 3; i++)
    f(Arr[i], Arr[i + (kNumArrays / 3)], Arr[i + (2 * kNumArrays / 3)],
      kArraySize);
}

// Takes a string of IR and compiles it using LLVM's JIT Engine
static void CreateAndRunJITFunc(const std::string &IR, CodeGenOpt::Level OLvl) {
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseIR(MemoryBufferRef(IR, "IR"), Err, Context);
  if (!M)
    ErrorAndExit("Could not parse IR");

  Function *EntryFunc = M->getFunction("foo");
  if (!EntryFunc)
    ErrorAndExit("Function not found in module");

  std::string ErrorMsg;
  Triple ModuleTriple(M->getTargetTriple());

  EngineBuilder builder(std::move(M));
  builder.setMArch(codegen::getMArch());
  builder.setMCPU(codegen::getCPUStr());
  builder.setMAttrs(codegen::getFeatureList());
  builder.setErrorStr(&ErrorMsg);
  builder.setEngineKind(EngineKind::JIT);
  builder.setMCJITMemoryManager(std::make_unique<SectionMemoryManager>());
  builder.setOptLevel(OLvl);
  builder.setTargetOptions(
      codegen::InitTargetOptionsFromCodeGenFlags(ModuleTriple));

  std::unique_ptr<ExecutionEngine> EE(builder.create());
  if (!EE)
    ErrorAndExit("Could not create execution engine");

  EE->finalizeObject();
  EE->runStaticConstructorsDestructors(false);

#if defined(__GNUC__) && !defined(__clang) &&                                  \
    ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9))
// Silence
//
//   warning: ISO C++ forbids casting between pointer-to-function and
//   pointer-to-object [-Wpedantic]
//
// Since C++11 this casting is conditionally supported and GCC versions
// starting from 4.9.0 don't warn about the cast.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  LLVMFunc f = reinterpret_cast<LLVMFunc>(EE->getPointerToFunction(EntryFunc));
#if defined(__GNUC__) && !defined(__clang) &&                                  \
    ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9))
#pragma GCC diagnostic pop
#endif

  // Figure out if we are running the optimized func or the unoptimized func
  RunFuncOnInputs(f, (OLvl == CodeGenOpt::None) ? UnoptArrays : OptArrays);

  EE->runStaticConstructorsDestructors(true);
}

// Main fuzz target called by ExampleClangLLVMProtoFuzzer.cpp
// Mimics the lli tool to JIT the LLVM IR code and execute it
void clang_fuzzer::HandleLLVM(const std::string &IR,
                              const std::vector<const char *> &ExtraArgs) {
  // Populate OptArrays and UnoptArrays with the arrays from InputArrays
  memcpy(OptArrays, InputArrays, kTotalSize);
  memcpy(UnoptArrays, InputArrays, kTotalSize);

  // Parse ExtraArgs to set the optimization level
  CodeGenOpt::Level OLvl;
  getOptLevel(ExtraArgs, OLvl);

  // First we optimize the IR by running a loop vectorizer pass
  std::string OptIR = OptLLVM(IR, OLvl);

  CreateAndRunJITFunc(OptIR, OLvl);
  CreateAndRunJITFunc(IR, CodeGenOpt::None);

  if (memcmp(OptArrays, UnoptArrays, kTotalSize))
    ErrorAndExit("!!!BUG!!!");

  return;
}
