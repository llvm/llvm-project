#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "complex-conversion"

namespace hlfir {
#define GEN_PASS_DEF_COMPLEXDIVISIONCONVERSION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} 

static llvm::cl::opt<bool> EnableArithmeticBasedComplexDiv(
    "enable-arithmetic-based-complex-div", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("Enable calling of Arithmetic-based Complex Division."));

namespace {
class HlfirComplexDivisionConversion : public mlir::OpRewritePattern<fir::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult matchAndRewrite(fir::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const override {
    if (!EnableArithmeticBasedComplexDiv) {
      LLVM_DEBUG(llvm::dbgs() << "Arithmetic-based Complex Division support is currently disabled \n");
      return mlir::failure();
    }
    fir::FirOpBuilder builder{rewriter, callOp.getOperation()};
    const mlir::Location &loc = callOp.getLoc();
    if (!callOp.getCallee()) {
      LLVM_DEBUG(llvm::dbgs() << "No callee found for CallOp at " << loc << "\n");
      return mlir::failure();
    } 

    const mlir::SymbolRefAttr &callee = *callOp.getCallee();
    const auto &fctName = callee.getRootReference().getValue();
    if (fctName!= "__divdc3")
      return mlir::failure();

    const mlir::Type &eleTy = callOp.getOperands()[0].getType();
    const mlir::Type &resTy = callOp.getResult(0).getType();

    auto x0 = callOp.getOperands()[0]; // real part of numerator : x0
    auto y0 = callOp.getOperands()[1]; // imaginary part of numerator : y0
    auto x1 = callOp.getOperands()[2]; // real part of denominator : x1
    auto y1 = callOp.getOperands()[3]; // imaginary part of denominator : y1

    // standard complex division formula:
    // (x0 + y0i)/(x1 + y1i) = ((x0*x1 + y0*y1)/(x1² + y1²)) + ((y0*x1 - x0*y1)/(x1² + y1²))i
    auto x0x1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x0, x1); // x0 * x1
    auto x1Squared = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x1, x1); // x1^2
    auto y0x1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y0, x1); // y0 * x1
    auto x0y1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x0, y1); // x0 * y1
    auto y0y1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y0, y1); // y0 * y1
    auto y1Squared = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y1, y1); // y1^2

    auto denom = rewriter.create<mlir::arith::AddFOp>(loc, eleTy, x1Squared, y1Squared); // x1^2 + y1^2
    auto realNumerator = rewriter.create<mlir::arith::AddFOp>(loc, eleTy, x0x1, y0y1); // x0*x1 + y0*y1
    auto imagNumerator = rewriter.create<mlir::arith::SubFOp>(loc, eleTy, y0x1, x0y1); // y0*x1 - x0*y1

    // compute final real and imaginary parts
    auto realResult = rewriter.create<mlir::arith::DivFOp>(loc, eleTy, realNumerator, denom);
    auto imagResult = rewriter.create<mlir::arith::DivFOp>(loc, eleTy, imagNumerator, denom);

    // construct the result complex number
    auto undefComplex = rewriter.create<fir::UndefOp>(loc, resTy);
    auto index0 = builder.getArrayAttr({builder.getI32IntegerAttr(0)}); // index for real part
    auto index1 = builder.getArrayAttr({builder.getI32IntegerAttr(1)}); // index for imag part
    auto complexWithReal = rewriter.create<fir::InsertValueOp>(loc, resTy, undefComplex, realResult, index0); // Insert real part
    auto resComplex = rewriter.create<fir::InsertValueOp>(loc, resTy, complexWithReal, imagResult, index1); // Insert imaginary part
    rewriter.replaceOp(callOp, resComplex.getResult());
    return mlir::success();
  }
};
class ComplexDivisionConversion : public hlfir::impl::ComplexDivisionConversionBase<ComplexDivisionConversion> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<HlfirComplexDivisionConversion>(context);
    
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled);

    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns), config)))
    {
      mlir::emitError(mlir::UnknownLoc::get(context), "failure in Arithmetic-based Complex Division HLFIR intrinsic lowering");
      signalPassFailure();
    }
  }
};
} 