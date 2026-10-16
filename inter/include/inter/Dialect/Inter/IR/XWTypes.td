// XW frontend types.

#ifndef INTER_DIALECT_INTER_IR_XWTYPES_TD
#define INTER_DIALECT_INTER_IR_XWTYPES_TD

include "inter/Dialect/Inter/IR/InterBase.td"
include "inter/Dialect/Inter/IR/XWAttrs.td"
include "mlir/IR/AttrTypeBase.td"

class XW_Type<string name, string typeMnemonic> : TypeDef<XWDialect, name> {
  let mnemonic = typeMnemonic;
}

def XW_SimdType : XW_Type<"Simd", "simd"> {
  let parameters = (ins "::mlir::Type":$elementType, "int64_t":$cardinality);
  let assemblyFormat = "`<` $elementType `,` $cardinality `>`";
  let genVerifyDecl = 1;
}
def XW_MaskType : XW_Type<"Mask", "mask"> {
  let parameters = (ins "int64_t":$cardinality);
  let assemblyFormat = "`<` $cardinality `>`";
  let genVerifyDecl = 1;
}
def XW_MemTokenType : XW_Type<"MemToken", "mem.token">;
def XW_PtrType : XW_Type<"Ptr", "ptr"> {
  let parameters = (ins "::mlir::Attribute":$addressSpace);
  let assemblyFormat = "`<` $addressSpace `>`";
  let genVerifyDecl = 1;
}

def XW_Simd : Type<CPred<"::llvm::isa<::xw::SimdType>($_self)">,
    "XW SIMD type">;
def XW_Mask : Type<CPred<"::llvm::isa<::xw::MaskType>($_self)">,
    "XW mask type">;
def XW_MemToken : Type<CPred<"::llvm::isa<::xw::MemTokenType>($_self)">,
    "XW memory token type">;
def XW_Ptr : Type<CPred<"::llvm::isa<::xw::PtrType>($_self)">,
    "XW pointer type">;
def XW_SimdOfPtr : Type<And<[XW_Simd.predicate,
    CPred<"::llvm::isa<::xw::PtrType>(::llvm::cast<::xw::SimdType>($_self).getElementType())">]>,
    "XW SIMD of pointers">;
def XW_PtrOrSimdPtr : Type<Or<[XW_Ptr.predicate, XW_SimdOfPtr.predicate]>,
    "XW pointer or SIMD of pointers">;
def XW_VectorOrSimdVector : Type<Or<[
    CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
    And<[XW_Simd.predicate,
      CPred<"::llvm::isa<::mlir::VectorType>(::llvm::cast<::xw::SimdType>($_self).getElementType())">]>
    ]>, "vector or XW SIMD of vector">;

#endif // INTER_DIALECT_INTER_IR_XWTYPES_TD
