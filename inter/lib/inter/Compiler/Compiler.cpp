#include "inter/Compiler/Compiler.h"

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Emit/Emit.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

namespace mlir {
std::unique_ptr<Pass> createLiftControlFlowToSCFPass();
}

namespace {

class InterTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<
          InterTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterTransformDialectExtension)

  using Base::Base;

  void init() {
    declareGeneratedDialect<xw::XWDialect>();
    declareGeneratedDialect<inter::xemachine::XeMachineDialect>();
    declareGeneratedDialect<mlir::func::FuncDialect>();
    declareGeneratedDialect<mlir::scf::SCFDialect>();
    declareGeneratedDialect<mlir::arith::ArithDialect>();
    declareGeneratedDialect<mlir::cf::ControlFlowDialect>();
    declareGeneratedDialect<mlir::ub::UBDialect>();
    declareGeneratedDialect<mlir::LLVM::LLVMDialect>();
    declareGeneratedDialect<mlir::DLTIDialect>();
    declareGeneratedDialect<mlir::memref::MemRefDialect>();
    declareGeneratedDialect<mlir::vector::VectorDialect>();
    declareGeneratedDialect<mlir::gpu::GPUDialect>();
    declareGeneratedDialect<mlir::xegpu::XeGPUDialect>();
    declareGeneratedDialect<mlir::xevm::XeVMDialect>();
  }
};

llvm::Error makeDiagnosticError(llvm::ArrayRef<std::string> diagnostics,
                                llvm::StringRef fallback) {
  if (diagnostics.empty())
    return llvm::createStringError(fallback);
  std::string message;
  llvm::raw_string_ostream stream(message);
  for (const std::string &diagnostic : diagnostics)
    stream << diagnostic << '\n';
  return llvm::createStringError(std::move(message));
}

} // namespace

void inter::registerCompilerDialects(mlir::DialectRegistry &registry) {
  registry.insert<xw::XWDialect, xemachine::XeMachineDialect,
                  mlir::transform::TransformDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::ub::UBDialect,
                  mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  mlir::gpu::GPUDialect, mlir::xegpu::XeGPUDialect,
                  mlir::xevm::XeVMDialect>();
  registry.addExtensions<InterTransformDialectExtension>();
}

void inter::registerCompilerPasses() {
  static std::once_flag once;
  std::call_once(once, [] {
    registerInterPasses();
    mlir::registerTransformsPasses();
    mlir::transform::registerTransformPasses();
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::createLiftControlFlowToSCFPass();
    });
  });
}

llvm::Error inter::compileLLVMModule(std::unique_ptr<llvm::Module> llvmModule,
                                     llvm::raw_ostream &output,
                                     llvm::raw_ostream &diagnosticOutput,
                                     const CompilerOptions &options) {
  if (!llvmModule)
    return llvm::createStringError("cannot compile a null LLVM module");
  if (options.transformLibraryPath.empty())
    return llvm::createStringError("Inter transform library path is empty");
  if (!options.target.supportsSimdWidth(options.simdWidth))
    return llvm::createStringError("SIMD width is unsupported by target '" +
                                   options.target.getChipName() + "'");

  registerCompilerPasses();
  mlir::DialectRegistry registry;
  registerCompilerDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SmallVector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        std::string &message = diagnostics.emplace_back();
        llvm::raw_string_ostream(message)
            << diagnostic.getLocation() << ": " << diagnostic;
        diagnosticOutput << message << '\n';
      });

  mlir::OwningOpRef<mlir::ModuleOp> mlirModule =
      mlir::translateLLVMIRToModule(std::move(llvmModule), &context);
  if (!mlirModule)
    return makeDiagnosticError(diagnostics, "LLVM IR import failed");

  mlirModule->getOperation()->setAttr(xemachine::kCompilationTargetAttrName,
                                      options.target.getAttr(&context));
  mlirModule->getOperation()->setAttr(
      xemachine::kCompilationSimdWidthAttrName,
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32),
                             options.simdWidth));

  mlir::PassManager manager(&context);
  mlir::transform::PreloadLibraryPassOptions preloadOptions;
  preloadOptions.transformLibraryPaths.push_back(options.transformLibraryPath);
  manager.addPass(
      mlir::transform::createPreloadLibraryPass(std::move(preloadOptions)));
  mlir::transform::InterpreterPassOptions interpreterOptions;
  interpreterOptions.entryPoint = "inter_backend";
  manager.addPass(
      mlir::transform::createInterpreterPass(std::move(interpreterOptions)));
  if (mlir::failed(manager.run(*mlirModule)))
    return makeDiagnosticError(diagnostics, "Inter lowering failed");

  mlir::LogicalResult result = mlir::success();
  switch (options.output) {
  case CompilationOutput::zebin:
    result = emitZebin(*mlirModule, output);
    break;
  case CompilationOutput::ged:
    result = emitGedBinary(*mlirModule, output);
    break;
  case CompilationOutput::assembly:
    result = emitAssembly(*mlirModule, output);
    break;
  case CompilationOutput::none:
    break;
  }
  if (mlir::failed(result))
    return makeDiagnosticError(diagnostics, "Inter emission failed");
  return llvm::Error::success();
}
