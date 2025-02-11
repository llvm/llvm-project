//===----------------------------------------------------------------------===//
// TBAAAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAAttr : CIR_Attr<"TBAA", "tbaa", []> {
  let summary = "CIR dialect TBAA base attribute";
}

//===----------------------------------------------------------------------===//
// TBAAOmnipotentCharAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAOmnipotentChar
    : CIR_Attr<"TBAAOmnipotentChar", "tbaa_omnipotent_char", [], "TBAAAttr"> {
  let summary = "Describes a special scalar type, the omnipotent char type.";
}

//===----------------------------------------------------------------------===//
// TBAAScalarAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAScalarAttr : CIR_Attr<"TBAAScalar", "tbaa_scalar", [], "TBAAAttr"> {
  let summary = "Describes a scalar type in TBAA with an identifier.";

  let parameters = (ins StringRefParameter<> : $id, CIR_AnyType : $type);

  let description = [{
    Define a TBAA scalar attribute.

    Example:
    ```mlir
    // CIR_TBAAScalarAttr
    #tbaa_scalar = #cir.tbaa_scalar<id = "int", type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<id = "long long", type = !s64i>
    ```
    
    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_TBAATagAttr : CIR_Attr<"TBAATag", "tbaa_tag", [], "TBAAAttr"> {
  let parameters = (ins CIR_TBAAAttr
                    : $base, CIR_TBAAAttr
                    : $access, "int64_t"
                    : $offset);

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_AnyTBAAAttr : AnyAttrOf<[
  CIR_TBAAAttr, 
  CIR_TBAAOmnipotentChar, 
  CIR_TBAAScalarAttr, 
  CIR_TBAATagAttr
]>;
