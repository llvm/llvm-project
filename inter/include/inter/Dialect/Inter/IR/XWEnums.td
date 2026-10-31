// XW frontend enums.

#ifndef INTER_DIALECT_INTER_IR_XWENUMS_TD
#define INTER_DIALECT_INTER_IR_XWENUMS_TD

include "inter/Dialect/Inter/IR/InterBase.td"
include "mlir/IR/EnumAttr.td"

def XW_CastKindFpConvert : I32EnumAttrCase<"FpConvert", 0, "fpconvert">;
def XW_CastKindIntConvert : I32EnumAttrCase<"IntConvert", 1, "intconvert">;
def XW_CastKindIntToFp : I32EnumAttrCase<"IntToFp", 2, "int_to_fp">;
def XW_CastKindFpToInt : I32EnumAttrCase<"FpToInt", 3, "fp_to_int">;
def XW_CastKindEnum : I32EnumAttr<"CastKind", "XW cast kind", [
    XW_CastKindFpConvert, XW_CastKindIntConvert, XW_CastKindIntToFp,
    XW_CastKindFpToInt]> {
  let cppNamespace = "::xw";
}

def XW_BinaryKindAddI : I32EnumAttrCase<"AddI", 0, "addi">;
def XW_BinaryKindSubI : I32EnumAttrCase<"SubI", 1, "subi">;
def XW_BinaryKindMulI : I32EnumAttrCase<"MulI", 2, "muli">;
def XW_BinaryKindShLI : I32EnumAttrCase<"ShLI", 3, "shli">;
def XW_BinaryKindShRUI : I32EnumAttrCase<"ShRUI", 4, "shrui">;
def XW_BinaryKindShRSI : I32EnumAttrCase<"ShRSI", 5, "shrsi">;
def XW_BinaryKindAndI : I32EnumAttrCase<"AndI", 6, "andi">;
def XW_BinaryKindOrI : I32EnumAttrCase<"OrI", 7, "ori">;
def XW_BinaryKindXOrI : I32EnumAttrCase<"XOrI", 8, "xori">;
def XW_BinaryKindDivUI : I32EnumAttrCase<"DivUI", 9, "divui">;
def XW_BinaryKindDivSI : I32EnumAttrCase<"DivSI", 10, "divsi">;
def XW_BinaryKindRemUI : I32EnumAttrCase<"RemUI", 11, "remui">;
def XW_BinaryKindRemSI : I32EnumAttrCase<"RemSI", 12, "remsi">;
def XW_BinaryKindMulHUI : I32EnumAttrCase<"MulHUI", 13, "mulhui">;
def XW_BinaryKindEnum : I32EnumAttr<"BinaryKind", "XW integer binary kind", [
    XW_BinaryKindAddI, XW_BinaryKindSubI, XW_BinaryKindMulI,
    XW_BinaryKindShLI, XW_BinaryKindShRUI, XW_BinaryKindShRSI,
    XW_BinaryKindAndI, XW_BinaryKindOrI, XW_BinaryKindXOrI,
    XW_BinaryKindDivUI, XW_BinaryKindDivSI, XW_BinaryKindRemUI,
    XW_BinaryKindRemSI, XW_BinaryKindMulHUI]> {
  let cppNamespace = "::xw";
}

def XW_CastRoundingRNE : I32EnumAttrCase<"RNE", 0, "rne">;
def XW_CastRoundingRTZ : I32EnumAttrCase<"RTZ", 1, "rtz">;
def XW_CastRoundingRTP : I32EnumAttrCase<"RTP", 2, "rtp">;
def XW_CastRoundingRTN : I32EnumAttrCase<"RTN", 3, "rtn">;
def XW_CastRoundingEnum : I32EnumAttr<"CastRounding", "XW cast rounding", [
    XW_CastRoundingRNE, XW_CastRoundingRTZ, XW_CastRoundingRTP,
    XW_CastRoundingRTN]> {
  let cppNamespace = "::xw";
}

def XW_CastSignednessSigned : I32EnumAttrCase<"Signed", 0, "signed">;
def XW_CastSignednessUnsigned : I32EnumAttrCase<"Unsigned", 1, "unsigned">;
def XW_CastSignednessEnum : I32EnumAttr<"CastSignedness",
    "XW cast integer signedness",
    [XW_CastSignednessSigned, XW_CastSignednessUnsigned]> {
  let cppNamespace = "::xw";
}

def XW_CastExtensionSign : I32EnumAttrCase<"Sign", 0, "sign">;
def XW_CastExtensionZero : I32EnumAttrCase<"Zero", 1, "zero">;
def XW_CastExtensionEnum : I32EnumAttr<"CastExtension",
    "XW cast integer extension", [XW_CastExtensionSign, XW_CastExtensionZero]> {
  let cppNamespace = "::xw";
}

def XW_CachePolicyUncached : I32EnumAttrCase<"Uncached", 0, "uncached">;
def XW_CachePolicyCached : I32EnumAttrCase<"Cached", 1, "cached">;
def XW_CachePolicyStreaming : I32EnumAttrCase<"Streaming", 2, "streaming">;
def XW_CachePolicyReadInvalidate
    : I32EnumAttrCase<"ReadInvalidate", 3, "read_invalidate">;
def XW_CachePolicyWriteBack
    : I32EnumAttrCase<"WriteBack", 4, "write_back">;
def XW_CachePolicyWriteThrough
    : I32EnumAttrCase<"WriteThrough", 5, "write_through">;
def XW_CachePolicyEnum : I32EnumAttr<"CachePolicyKind", "XW cache policy", [
    XW_CachePolicyUncached, XW_CachePolicyCached, XW_CachePolicyStreaming,
    XW_CachePolicyReadInvalidate, XW_CachePolicyWriteBack,
    XW_CachePolicyWriteThrough]> {
  let cppNamespace = "::xw";
}

def XW_DpasPrecisionF16 : I32EnumAttrCase<"F16", 0, "f16">;
def XW_DpasPrecisionBF16 : I32EnumAttrCase<"BF16", 1, "bf16">;
def XW_DpasPrecisionEnum : I32EnumAttr<"DpasPrecision",
    "DPAS source precision", [XW_DpasPrecisionF16, XW_DpasPrecisionBF16]> {
  let cppNamespace = "::xw";
}

#endif // INTER_DIALECT_INTER_IR_XWENUMS_TD
