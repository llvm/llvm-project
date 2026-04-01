//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Func/Extensions/AllExtensions.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/AIIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "aiir/Dialect/Affine/Transforms/Passes.h"
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h"
#include "aiir/ExecutionEngine/ExecutionEngine.h"
#include "aiir/ExecutionEngine/OptUtils.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Verifier.h"
#include "aiir/InitAllDialects.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Export.h"
#include "aiir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, AIIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(AIIR, "aiir",
                          "load the input file as an AIIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpAIIR,
  DumpAIIRAffine,
  DumpAIIRLLVM,
  DumpLLVMIR,
  RunJIT
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpAIIR, "aiir", "output the AIIR dump")),
    cl::values(clEnumValN(DumpAIIRAffine, "aiir-affine",
                          "output the AIIR dump after affine lowering")),
    cl::values(clEnumValN(DumpAIIRLLVM, "aiir-llvm",
                          "output the AIIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<toy::ModuleAST>
parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

static int loadAIIR(aiir::AIIRContext &context,
                    aiir::OwningOpRef<aiir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::AIIR &&
      !llvm::StringRef(inputFilename).ends_with(".aiir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = aiirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.aiir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input aiir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = aiir::parseSourceFile<aiir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int loadAndProcessAIIR(aiir::AIIRContext &context,
                              aiir::OwningOpRef<aiir::ModuleOp> &module) {
  if (int error = loadAIIR(context, module))
    return error;

  aiir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (aiir::failed(aiir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of AIIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpAIIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpAIIRLLVM;

  if (enableOpt || isLoweringToAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(aiir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    aiir::OpPassManager &optPM = pm.nest<aiir::toy::FuncOp>();
    optPM.addPass(aiir::createCanonicalizerPass());
    optPM.addPass(aiir::toy::createShapeInferencePass());
    optPM.addPass(aiir::createCanonicalizerPass());
    optPM.addPass(aiir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect.
    pm.addPass(aiir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    aiir::OpPassManager &optPM = pm.nest<aiir::func::FuncOp>();
    optPM.addPass(aiir::createCanonicalizerPass());
    optPM.addPass(aiir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(aiir::affine::createLoopFusionPass());
      optPM.addPass(aiir::affine::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(aiir::toy::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(aiir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (aiir::failed(pm.run(*module)))
    return 4;
  return 0;
}

static int dumpAST() {
  if (inputType == InputType::AIIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is AIIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

static int dumpLLVMIR(aiir::ModuleOp module) {
  // Register the translation to LLVM IR with the AIIR context.
  aiir::registerBuiltinDialectTranslation(*module->getContext());
  aiir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = aiir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  aiir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = aiir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static int runJit(aiir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from AIIR to LLVM IR, which must happen before we
  // can JIT-compile.
  aiir::registerBuiltinDialectTranslation(*module->getContext());
  aiir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = aiir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an AIIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  aiir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = aiir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  aiir::registerAsmPrinterCLOptions();
  aiir::registerAIIRContextCLOptions();
  aiir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to AIIR.
  aiir::DialectRegistry registry;
  aiir::func::registerAllExtensions(registry);
  aiir::LLVM::registerInlinerInterface(registry);

  aiir::AIIRContext context(registry);
  // Load our Dialect in this AIIR Context.
  context.getOrLoadDialect<aiir::toy::ToyDialect>();

  aiir::OwningOpRef<aiir::ModuleOp> module;
  if (int error = loadAndProcessAIIR(context, module))
    return error;

  // If we aren't exporting to non-aiir, then we are done.
  bool isOutputingAIIR = emitAction <= Action::DumpAIIRLLVM;
  if (isOutputingAIIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
