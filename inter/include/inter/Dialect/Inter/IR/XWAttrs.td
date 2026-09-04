// XW frontend attributes.

#ifndef INTER_DIALECT_INTER_IR_XWATTRS_TD
#define INTER_DIALECT_INTER_IR_XWATTRS_TD

include "inter/Dialect/Inter/IR/InterBase.td"
include "inter/Dialect/Inter/IR/XWEnums.td"
include "mlir/IR/AttrTypeBase.td"

class XW_Attr<string name, string attrMnemonic>
    : AttrDef<XWDialect, name> {
  let mnemonic = attrMnemonic;
}

def XW_CastRoundingPolicyAttr : XW_Attr<"CastRoundingPolicy", "cast_rounding"> {
  let parameters = (ins EnumParameter<XW_CastRoundingEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}
def XW_CastSignednessPolicyAttr
    : XW_Attr<"CastSignednessPolicy", "cast_signedness"> {
  let parameters = (ins EnumParameter<XW_CastSignednessEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}
def XW_CastExtensionPolicyAttr
    : XW_Attr<"CastExtensionPolicy", "cast_extension"> {
  let parameters = (ins EnumParameter<XW_CastExtensionEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}
def XW_CachePolicyAttr : XW_Attr<"CachePolicy", "cache_policy"> {
  let parameters = (ins EnumParameter<XW_CachePolicyEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def XW_PrivateAddressSpaceAttr : XW_Attr<"PrivateAddressSpace", "private"> {
  let assemblyFormat = "";
}
def XW_GlobalAddressSpaceAttr : XW_Attr<"GlobalAddressSpace", "global"> {
  let assemblyFormat = "";
}
def XW_ConstantAddressSpaceAttr : XW_Attr<"ConstantAddressSpace", "constant"> {
  let assemblyFormat = "";
}
def XW_LocalAddressSpaceAttr : XW_Attr<"LocalAddressSpace", "local"> {
  let assemblyFormat = "";
}
def XW_GenericAddressSpaceAttr : XW_Attr<"GenericAddressSpace", "generic"> {
  let assemblyFormat = "";
}

#endif // INTER_DIALECT_INTER_IR_XWATTRS_TD
