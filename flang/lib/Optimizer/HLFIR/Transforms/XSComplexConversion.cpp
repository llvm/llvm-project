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
#define DEBUG_TYPE "xs-complex-conversion"

namespace hlfir {
#define GEN_PASS_DEF_XSCOMPLEXCONVERSION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} 

static llvm::cl::opt<bool> EnableXSDivc(
    "enable-XSDivc", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("Enable calling of XSDivc."));

namespace {
class HlfirXSComplexConversion : public mlir::OpRewritePattern<fir::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult matchAndRewrite(fir::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const override {
    if (!EnableXSDivc) {
      LLVM_DEBUG(llvm::dbgs() << "XS Complex Division support is currently disabled \n");
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

    auto x0 = callOp.getOperands()[0];
    auto y0 = callOp.getOperands()[1];
    auto x1 = callOp.getOperands()[2];
    auto y1 = callOp.getOperands()[3];

    auto xx = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x0, x1);
    auto x1x1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x1, x1);
    auto yx = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, x0, y1);
    auto yy = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y0, y1);
    auto y1y1 = rewriter.create<mlir::arith::MulFOp>(loc, eleTy, y1, y1);
    auto d = rewriter.create<mlir::arith::AddFOp>(loc, eleTy, x1x1, y1y1);
    auto rrn = rewriter.create<mlir::arith::AddFOp>(loc, eleTy, xx, yy);
    auto rin = rewriter.create<mlir::arith::SubFOp>(loc, eleTy, yx, xy);
    auto rr = rewriter.create<mlir::arith::DivFOp>(loc, eleTy, rrn, d);
    auto ri = rewriter.create<mlir::arith::DivFOp>(loc, eleTy, rin, d);
    auto ra = rewriter.create<fir::UndefOp>(loc, resTy);
    auto indexAttr0 = builder.getArrayAttr({builder.getI32IntegerAttr(0)});
    auto indexAttr1 = builder.getArrayAttr({builder.getI32IntegerAttr(1)});
    auto r1 = rewriter.create<fir::InsertValueOp>(loc,resTy, ra, rr, indexAttr0);
    auto r0 = rewriter.create<fir::InsertValueOp>(loc,resTy, r1, ri, indexAttr1);
    rewriter.replaceOp(callOp, r0.getResult());
    return mlir::success();
  }
};

class XSComplexConversion : public hlfir::impl::XSComplexConversionBase<XSComplexConversion> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<HlfirXSComplexConversion>(context);
    
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled);

    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns), config)))
    {
      mlir::emitError(mlir::UnknownLoc::get(context), "failure in XS Complex HLFIR intrinsic lowering");
      signalPassFailure();
    }
  }
};
} 