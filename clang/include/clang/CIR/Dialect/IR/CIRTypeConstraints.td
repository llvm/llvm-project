//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIR dialect type constraints.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPECONSTRAINTS_TD
#define CLANG_CIR_DIALECT_IR_CIRTYPECONSTRAINTS_TD

include "mlir/IR/Constraints.td"
include "mlir/IR/CommonTypeConstraints.td"

class CIR_IsTypePred<code type> : CPred<"::mlir::isa<" # type # ">($_self)">;

class CIR_TypeBase<code type, string summary = "">
    : Type<CIR_IsTypePred<type>, summary, type>;

class CIR_CastSelfToType<code type, Pred pred>
    : SubstLeaves<"$_self", "::mlir::cast<" # type # ">($_self)", pred>;

class CIR_CastedSelfsToType<code type, list<Pred> preds>
    : And<!foreach(pred, preds, CIR_CastSelfToType<type, pred>)>;

class CIR_ConfinedType<Type type, list<Pred> preds, string summary = "">
    : Type<And<[type.predicate, CIR_CastedSelfsToType<type.cppType, preds>]>,
         summary, type.cppType>;

// Generates a type summary.
// - For a single type: returns its summary.
// - For multiple types: returns `any of <comma-separated summaries>`.
class CIR_TypeSummaries<list<Type> types> {
    assert !not(!empty(types)), "expects non-empty list of types";

    list<string> summaries = !foreach(type, types, type.summary);
    string joined = !interleave(summaries, ", ");

    string value = !if(!eq(!size(types), 1), joined, "any of " # joined);
}

//===----------------------------------------------------------------------===//
// Bool Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyBoolType : CIR_TypeBase<"::cir::BoolType", "boolean type">;

//===----------------------------------------------------------------------===//
// IntType predicates
//===----------------------------------------------------------------------===//

def CIR_AnyIntType : CIR_TypeBase<"::cir::IntType", "integer type">;

def CIR_AnyUIntType : CIR_ConfinedType<CIR_AnyIntType, [
    CPred<"$_self.isUnsigned()">], "unsigned integer type">;

def CIR_AnySIntType : CIR_ConfinedType<CIR_AnyIntType, [
    CPred<"$_self.isSigned()">], "signed integer type">;

class CIR_HasWidthPred<int width> : CPred<"$_self.getWidth() == " # width>;

def CIR_HasFundamentalIntWidthPred
    : CPred<"::cir::isValidFundamentalIntWidth($_self.getWidth())">;

class CIR_IntOfWidthsPred<list<int> widths>
    : Or<!foreach(width, widths, CIR_HasWidthPred<width>)>;

class CIR_IntOfWidths<list<int> widths>
    : CIR_ConfinedType<CIR_AnyIntType, [CIR_IntOfWidthsPred<widths>],
        "integer type of widths " # !interleave(widths, "/")>;

class CIR_SIntOfWidths<list<int> widths>
    : CIR_ConfinedType<CIR_AnySIntType, [CIR_IntOfWidthsPred<widths>],
        "signed integer type of widths " # !interleave(widths, "/")>;

class CIR_UIntOfWidths<list<int> widths>
    : CIR_ConfinedType<CIR_AnyUIntType, [CIR_IntOfWidthsPred<widths>],
        "unsigned integer type of widths " # !interleave(widths, "/")>;

class CIR_UInt<int width>
    : CIR_ConfinedType<CIR_AnyUIntType, [CIR_HasWidthPred<width>],
        width # "-bit unsigned integer">,
      BuildableType<"$_builder.getType<" # cppType # ">(" #
        width # ", /*isSigned=*/false)">;

def CIR_UInt1 : CIR_UInt<1>;
def CIR_UInt8 : CIR_UInt<8>;
def CIR_UInt16 : CIR_UInt<16>;
def CIR_UInt32 : CIR_UInt<32>;
def CIR_UInt64 : CIR_UInt<64>;
def CIR_UInt128 : CIR_UInt<128>;

class CIR_SInt<int width>
    : CIR_ConfinedType<CIR_AnySIntType, [CIR_HasWidthPred<width>],
        width # "-bit signed integer">,
      BuildableType<"$_builder.getType<" # cppType # ">(" #
        width # ", /*isSigned=*/true)">;

def CIR_SInt1 : CIR_SInt<1>;
def CIR_SInt8 : CIR_SInt<8>;
def CIR_SInt16 : CIR_SInt<16>;
def CIR_SInt32 : CIR_SInt<32>;
def CIR_SInt64 : CIR_SInt<64>;
def CIR_SInt128 : CIR_SInt<128>;

// Fundamental integer types represent standard source-level integer types that
// have a specified set of admissible bitwidths (8, 16, 32, 64).

def CIR_AnyFundamentalIntType
    : CIR_ConfinedType<CIR_AnyIntType, [CIR_HasFundamentalIntWidthPred],
        "fundamental integer type"> {
    let cppFunctionName = "isFundamentalIntType";
}

def CIR_AnyFundamentalUIntType
    : CIR_ConfinedType<CIR_AnyUIntType, [CIR_HasFundamentalIntWidthPred],
        "fundamental unsigned integer type"> {
    let cppFunctionName = "isFundamentalUIntType";
}

def CIR_AnyFundamentalSIntType
    : CIR_ConfinedType<CIR_AnySIntType, [CIR_HasFundamentalIntWidthPred],
        "fundamental signed integer type"> {
    let cppFunctionName = "isFundamentalSIntType";
}

//===----------------------------------------------------------------------===//
// Float Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnySingleType : CIR_TypeBase<"::cir::SingleType", "single float type">;
def CIR_AnyFP32Type : TypeAlias<CIR_AnySingleType>;

def CIR_AnyDoubleType : CIR_TypeBase<"::cir::DoubleType", "double float type">;
def CIR_AnyFP64Type : TypeAlias<CIR_AnyDoubleType>;

def CIR_AnyFP16Type : CIR_TypeBase<"::cir::FP16Type", "f16 type">;
def CIR_AnyBFloat16Type : CIR_TypeBase<"::cir::BF16Type", "bf16 type">;
def CIR_AnyFP80Type : CIR_TypeBase<"::cir::FP80Type", "f80 type">;
def CIR_AnyFP128Type : CIR_TypeBase<"::cir::FP128Type", "f128 type">;
def CIR_AnyLongDoubleType : CIR_TypeBase<"::cir::LongDoubleType",
    "long double type">;

def CIR_AnyFloatType : AnyTypeOf<[
    CIR_AnySingleType, CIR_AnyDoubleType, CIR_AnyFP16Type,
    CIR_AnyBFloat16Type, CIR_AnyFP80Type, CIR_AnyFP128Type,
    CIR_AnyLongDoubleType
]> {
    let cppFunctionName = "isAnyFloatingPointType";
}

def CIR_AnyIntOrFloatType : AnyTypeOf<[CIR_AnyFloatType, CIR_AnyIntType],
    "integer or floating point type"
> {
    let cppFunctionName = "isAnyIntegerOrFloatingPointType";
}

def CIR_AnyIntOrBoolOrFloatType
    : AnyTypeOf<[CIR_AnyBoolType, CIR_AnyFloatType, CIR_AnyIntType],
                "integer, boolean or floating point type"> {
  let cppFunctionName = "isAnyIntegerOrBooleanOrFloatingPointType";
}

//===----------------------------------------------------------------------===//
// Complex Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyComplexType : CIR_TypeBase<"::cir::ComplexType", "complex type">;

def CIR_AnyComplexOrIntOrBoolOrFloatType
    : AnyTypeOf<[CIR_AnyComplexType, CIR_AnyIntOrBoolOrFloatType],
                "complex, integer or floating point type"> {
  let cppFunctionName = "isComplexOrIntegerOrBoolOrFloatingPointType";
}

//===----------------------------------------------------------------------===//
// Record Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyRecordType : CIR_TypeBase<"::cir::RecordType", "record type">;

//===----------------------------------------------------------------------===//
// Array Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyArrayType : CIR_TypeBase<"::cir::ArrayType", "array type">;

//===----------------------------------------------------------------------===//
// Pointer Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyPtrType : CIR_TypeBase<"::cir::PointerType", "pointer type">;

// Pointer to type constraint bases
class CIR_IsPtrToPred<code type> : CPred<"$_self.isPtrTo<" # type # ">()">;

class CIR_PtrTo<code type, string summary>
    : CIR_ConfinedType<CIR_AnyPtrType, [CIR_IsPtrToPred<type>],
        "pointer to " # summary>;

// Pointer to pointer constraint bases
class CIR_IsPtrToPtrToPred<code type>
    : CPred<"$_self.isPtrToPtrTo<" # type # ">()">;

class CIR_PtrToPtrTo<code type, string summary>
    : CIR_ConfinedType<CIR_AnyPtrType, [CIR_IsPtrToPtrToPred<type>],
        "pointer to pointer to " # summary>;

// Pointee type constraint bases
class CIR_PointeePred<Pred pred> : SubstLeaves<"$_self",
      "::mlir::cast<::cir::PointerType>($_self).getPointee()", pred>;

class CIR_PtrToAnyOf<list<Type> types, string summary = "">
: CIR_ConfinedType<CIR_AnyPtrType,
  [Or<!foreach(type, types, CIR_PointeePred<type.predicate>)>],
  !if(!empty(summary),
      "pointer to " # CIR_TypeSummaries<types>.value,
      summary)>;

// Void pointer type constraints
def CIR_VoidPtrType
    : CIR_PtrTo<"::cir::VoidType", "void type">,
      BuildableType<"$_builder.getType<" # cppType # ">("
        "cir::VoidType::get($_builder.getContext()))">;

def CIR_PtrToVoidPtrType
    : CIR_PtrToPtrTo<"::cir::VoidType", "void type">,
      BuildableType<"$_builder.getType<" # cppType # ">("
        "$_builder.getType<" # cppType # ">("
        "cir::VoidType::get($_builder.getContext())))">;

class CIR_PtrToType<Type type> : CIR_PtrToAnyOf<[type]>;

// Pointer to type constraints
def CIR_PtrToIntOrFloatType : CIR_PtrToType<CIR_AnyIntOrFloatType>;

def CIR_PtrToComplexType : CIR_PtrToType<CIR_AnyComplexType>;

def CIR_PtrToRecordType : CIR_PtrToType<CIR_AnyRecordType>;

def CIR_PtrToArray : CIR_PtrToType<CIR_AnyArrayType>;

//===----------------------------------------------------------------------===//
// Vector Type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyVectorType : CIR_TypeBase<"::cir::VectorType", "vector type">;

def CIR_VectorElementType : AnyTypeOf<[CIR_AnyIntOrFloatType, CIR_AnyPtrType],
    "any cir integer, floating point or pointer type"
> {
    let cppFunctionName = "isValidVectorTypeElementType";
}

class CIR_ElementTypePred<Pred pred> : SubstLeaves<"$_self",
    "::mlir::cast<::cir::VectorType>($_self).getElementType()", pred>;

class CIR_VectorTypeOf<list<Type> types, string summary = "">
    : CIR_ConfinedType<CIR_AnyVectorType,
        [Or<!foreach(type, types, CIR_ElementTypePred<type.predicate>)>],
        !if(!empty(summary),
            "vector of " # CIR_TypeSummaries<types>.value,
            summary)>;

// Vector of integral type
def IntegerVector : Type<
    And<[
      CPred<"::mlir::isa<::cir::VectorType>($_self)">,
      CPred<"::mlir::isa<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getElementType())">,
      CPred<"::mlir::cast<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getElementType())"
            ".isFundamental()">
    ]>, "!cir.vector of !cir.int"> {
}

// Vector of type constraints
def CIR_VectorOfIntType : CIR_VectorTypeOf<[CIR_AnyIntType]>;
def CIR_VectorOfUIntType : CIR_VectorTypeOf<[CIR_AnyUIntType]>;
def CIR_VectorOfSIntType : CIR_VectorTypeOf<[CIR_AnySIntType]>;
def CIR_VectorOfFloatType : CIR_VectorTypeOf<[CIR_AnyFloatType]>;

// Vector or Scalar type constraints
def CIR_AnyIntOrVecOfIntType
    : AnyTypeOf<[CIR_AnyIntType, CIR_VectorOfIntType],
        "integer or vector of integer type"> {
    let cppFunctionName = "isIntOrVectorOfIntType";
}

def CIR_AnySIntOrVecOfSIntType
    : AnyTypeOf<[CIR_AnySIntType, CIR_VectorOfSIntType],
        "signed integer or vector of signed integer type"> {
    let cppFunctionName = "isSIntOrVectorOfSIntType";
}

def CIR_AnyUIntOrVecOfUIntType
    : AnyTypeOf<[CIR_AnyUIntType, CIR_VectorOfUIntType],
        "unsigned integer or vector of unsigned integer type"> {
    let cppFunctionName = "isUIntOrVectorOfUIntType";
}

def CIR_AnyFloatOrVecOfFloatType
    : AnyTypeOf<[CIR_AnyFloatType, CIR_VectorOfFloatType],
        "floating point or vector of floating point type"> {
    let cppFunctionName = "isFPOrVectorOfFPType";
}

//===----------------------------------------------------------------------===//
// VPtr type predicates
//===----------------------------------------------------------------------===//

def CIR_AnyVPtrType : CIR_TypeBase<"::cir::VPtrType", "vptr type">;

def CIR_PtrToVPtr : CIR_PtrToType<CIR_AnyVPtrType>;

//===----------------------------------------------------------------------===//
// Scalar Type predicates
//===----------------------------------------------------------------------===//

defvar CIR_ScalarTypes = [
    CIR_AnyBoolType, CIR_AnyIntType, CIR_AnyFloatType, CIR_AnyPtrType
];

def CIR_AnyScalarType : AnyTypeOf<CIR_ScalarTypes, "cir scalar type"> {
  let cppFunctionName = "isScalarType";
}

#endif // CLANG_CIR_DIALECT_IR_CIRTYPECONSTRAINTS_TD
