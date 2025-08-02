// RUN: clang-tblgen -gen-cir-builtins-lowering -I%S %s -o - 2>&1 | \
// RUN:    FileCheck --strict-whitespace %s
// REQUIRES: clangir

def ins;
def outs;

class OpBase {
  dag arguments = (ins);
  dag results = (outs);
}

class LLVMLoweringInfo {
  string llvmOp = "";
}

class CIR_Op : LLVMLoweringInfo, OpBase;

def CIR_FAbsOp : CIR_Op {
  let arguments = (ins);
  let results = (outs);

  let llvmOp = "FAbsOp";
}

// CHECK: #ifdef GET_BUILTIN_LOWERING_CLASSES_DECLARE
// CHECK: class CIRFAbsOpLowering : public mlir::OpConversionPattern<cir::FAbsOp> {
// CHECK:   public:
// CHECK:     using OpConversionPattern<cir::FAbsOp>::OpConversionPattern;
// CHECK:   mlir::LogicalResult
// CHECK:   matchAndRewrite(cir::FAbsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override;
// CHECK: };
// CHECK: #endif
// CHECK: #ifdef GET_BUILTIN_LOWERING_CLASSES_DEF
// CHECK: mlir::LogicalResult
// CHECK: CIRFAbsOpLowering::matchAndRewrite(cir::FAbsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {
// CHECK:   rewriter.replaceOpWithNewOp<mlir::LLVM::FAbsOp>(op);
// CHECK:   return mlir::success();
// CHECK: }
// CHECK: #endif
// CHECK: #ifdef GET_BUILTIN_LOWERING_LIST
// CHECK: , CIRFAbsOpLowering
// CHECK: #endif
