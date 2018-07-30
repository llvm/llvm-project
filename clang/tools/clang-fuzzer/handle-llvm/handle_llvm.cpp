//==-- handle_llvm.cpp - Helper function for Clang fuzzers -----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

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

void ErrorAndExit(std::string message) {
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
std::string OptLLVM(const std::string &IR, CodeGenOpt::Level OLvl) {
  // Create a module that will run the optimization passes
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseIR(MemoryBufferRef(IR, "IR"), Err, Context);
  if (!M || verifyModule(*M, &errs()))
    ErrorAndExit("Could not parse IR");

  setFunctionAttributes(getCPUStr(), getFeaturesStr(), *M);
  
  legacy::PassManager Passes;
  Triple ModuleTriple(M->getTargetTriple());
  
  Passes.add(new TargetLibraryInfoWrapperPass(ModuleTriple));
  Passes.add(createTargetTransformInfoWrapperPass(TargetIRAnalysis()));
  Passes.add(createVerifierPass());

  AddOptimizationPasses(Passes, OLvl, 0);
  
  // Add a pass that writes the optimized IR to an output stream
  std::string outString;
  raw_string_ostream OS(outString);
  Passes.add(createPrintModulePass(OS, "", false));

  Passes.run(*M);

  return OS.str();
}

void CreateAndRunJITFun(const std::string &IR, CodeGenOpt::Level OLvl) {
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseIR(MemoryBufferRef(IR, "IR"), Err,
                                          Context);
  if (!M)
    ErrorAndExit("Could not parse IR");

  Function *EntryFunc = M->getFunction("foo");
  if (!EntryFunc)
    ErrorAndExit("Function not found in module");

  std::string ErrorMsg;
  EngineBuilder builder(std::move(M));
  builder.setMArch(MArch);
  builder.setMCPU(getCPUStr());
  builder.setMAttrs(getFeatureList());
  builder.setErrorStr(&ErrorMsg);
  builder.setEngineKind(EngineKind::JIT);
  builder.setUseOrcMCJITReplacement(false);
  builder.setMCJITMemoryManager(make_unique<SectionMemoryManager>());
  builder.setOptLevel(OLvl);
  builder.setTargetOptions(InitTargetOptionsFromCodeGenFlags());

  std::unique_ptr<ExecutionEngine> EE(builder.create());
  if (!EE)
    ErrorAndExit("Could not create execution engine");

  EE->finalizeObject();
  EE->runStaticConstructorsDestructors(false);

  typedef void (*func)(int*, int*, int*, int);
  func f = reinterpret_cast<func>(EE->getPointerToFunction(EntryFunc)); 

  // Define some dummy arrays to use an input for now
  int a[] = {1};
  int b[] = {1};
  int c[] = {1};
  f(a, b, c, 1);

  EE->runStaticConstructorsDestructors(true);
}

// Main fuzz target called by ExampleClangLLVMProtoFuzzer.cpp
// Mimics the lli tool to JIT the LLVM IR code and execute it
void clang_fuzzer::HandleLLVM(const std::string &IR,
                              const std::vector<const char *> &ExtraArgs) {
  // Parse ExtraArgs to set the optimization level
  CodeGenOpt::Level OLvl;
  getOptLevel(ExtraArgs, OLvl);

  // First we optimize the IR by running a loop vectorizer pass
  std::string OptIR = OptLLVM(IR, OLvl);

  CreateAndRunJITFun(OptIR, OLvl);
  CreateAndRunJITFun(IR, CodeGenOpt::None);
  
  return;
}
