// The closed XW frontend dialect.

#ifndef INTER_DIALECT_INTER_IR_INTERBASE_TD
#define INTER_DIALECT_INTER_IR_INTERBASE_TD

include "mlir/IR/DialectBase.td"

def XWDialect : Dialect {
  let name = "xw";
  let summary = "Closed SIMD kernel frontend dialect";
  let description = [{
    XW models uniform builtin values, explicitly lane-varying SIMD values,
    masks, opaque pointers, structured predication, and explicit memory
    dependency tokens without depending on a target or lowering dialect.
  }];
  let cppNamespace = "::xw";
  let dependentDialects = ["arith::ArithDialect"];
  let hasConstantMaterializer = 1;
  let hasOperationAttrVerify = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral getSimdWidthAttrName() {
      return "xw.simd_width";
    }
  }];
}

#endif // INTER_DIALECT_INTER_IR_INTERBASE_TD
