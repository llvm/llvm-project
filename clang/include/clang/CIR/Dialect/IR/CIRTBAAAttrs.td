//===----------------------------------------------------------------------===//
// TBAAAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAAttr : CIR_Attr<"TBAA", "tbaa", []> {
  let summary = "CIR dialect TBAA base attribute";
}

//===----------------------------------------------------------------------===//
// TBAAScalarAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAScalarAttr : CIR_Attr<"TBAAScalar", "tbaa_scalar", [], "TBAAAttr"> {
  let summary = "Describes a scalar type in TBAA with an identifier.";

  let parameters = (ins CIR_AnyScalarType : $type);

  let description = [{
    Define a TBAA attribute.

    Example:
    ```mlir
    // CIR_TBAAScalarAttr
    #tbaa_scalar = #cir.tbaa_scalar<type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<type = !u32i>
    ```
    
    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_AnyTBAAAttr : AnyAttrOf<[
  CIR_TBAAAttr,
  CIR_TBAAScalarAttr
]>;
