// inter-translate: file-format crossings for the inter pipeline.
// --import-llvm:        LLVM IR -> MLIR llvm dialect
// --xemachine-to-ged:   xemachine MLIR -> raw kernel bytes
// --xemachine-to-asm:   xemachine MLIR -> final assembly text
// --xemachine-to-zebin: xemachine MLIR -> runnable zebin

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Emit/Emit.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));
static cl::opt<std::string> outputFilename("o", cl::desc("output file"),
                                           cl::init("-"));
static cl::opt<bool> importLLVM("import-llvm",
                                cl::desc("import LLVM IR to the llvm dialect"),
                                cl::init(false));
static cl::opt<bool> toGed("xemachine-to-ged",
                           cl::desc("emit raw kernel bytes via GED"),
                           cl::init(false));
static cl::opt<bool>
    toAsm("xemachine-to-asm",
          cl::desc("emit final assembly text via IGA's GED decoder"),
          cl::init(false));
static cl::opt<bool> toZebin("xemachine-to-zebin",
                             cl::desc("emit a runnable zebin"),
                             cl::init(false));
} // namespace llvm

static std::unique_ptr<llvm::ToolOutputFile> openOutput() {
  std::error_code ec;
  auto out = std::make_unique<llvm::ToolOutputFile>(llvm::outputFilename, ec,
                                                    llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "error: " << ec.message() << "\n";
    return nullptr;
  }
  return out;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "inter translate tool\n");

  unsigned translations = (llvm::importLLVM ? 1u : 0u) +
                          (llvm::toGed ? 1u : 0u) + (llvm::toAsm ? 1u : 0u) +
                          (llvm::toZebin ? 1u : 0u);
  if (translations != 1) {
    llvm::errs() << "error: select exactly one translation\n";
    return 1;
  }

  if (llvm::importLLVM) {
    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> llvmMod =
        llvm::parseIRFile(llvm::inputFilename, err, llvmContext);
    if (!llvmMod) {
      err.print(argv[0], llvm::errs());
      return 1;
    }
    mlir::DialectRegistry registry;
    registry.insert<mlir::DLTIDialect, mlir::LLVM::LLVMDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    mlir::OwningOpRef<mlir::ModuleOp> mod =
        mlir::translateLLVMIRToModule(std::move(llvmMod), &context);
    if (!mod) {
      llvm::errs() << "error: LLVM IR import failed\n";
      return 1;
    }
    auto out = openOutput();
    if (!out)
      return 1;
    mod->print(out->os());
    out->keep();
    return 0;
  }

  if (llvm::toGed || llvm::toAsm || llvm::toZebin) {
    mlir::DialectRegistry registry;
    registry.insert<inter::xemachine::XeMachineDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::DLTIDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto mod =
        mlir::parseSourceFile<mlir::ModuleOp>(llvm::inputFilename, &context);
    if (!mod) {
      llvm::errs() << "error: MLIR parse failed\n";
      return 1;
    }
    auto out = openOutput();
    if (!out)
      return 1;
    mlir::LogicalResult result = mlir::failure();
    if (llvm::toZebin)
      result = inter::emitZebin(mod.get(), out->os());
    else if (llvm::toAsm)
      result = inter::emitAssembly(mod.get(), out->os());
    else
      result = inter::emitGedBinary(mod.get(), out->os());
    if (mlir::failed(result))
      return 1;
    out->keep();
    return 0;
  }

  llvm_unreachable("translation selection validated above");
}
