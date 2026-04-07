//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend generates hlsl_alias_intrinsics_gen.inc (alias
// overloads) and hlsl_inline_intrinsics_gen.inc (inline/detail overloads) for
// HLSL intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

/// Minimum shader model version that supports 16-bit types.
static constexpr StringLiteral SM6_2 = "6.2";

//===----------------------------------------------------------------------===//
// Type name helpers
//===----------------------------------------------------------------------===//

static std::string getVectorTypeName(StringRef ElemType, unsigned N) {
  return (ElemType + Twine(N)).str();
}

static std::string getMatrixTypeName(StringRef ElemType, unsigned Rows,
                                     unsigned Cols) {
  return (ElemType + Twine(Rows) + "x" + Twine(Cols)).str();
}

/// Get the fixed type name string for a VectorType or HLSLType record.
static std::string getFixedTypeName(const Record *R) {
  if (R->isSubClassOf("VectorType"))
    return getVectorTypeName(
        R->getValueAsDef("ElementType")->getValueAsString("Name"),
        R->getValueAsInt("Size"));
  assert(R->isSubClassOf("HLSLType"));
  return R->getValueAsString("Name").str();
}

/// For a VectorType, return its ElementType record; for an HLSLType, return
/// the record itself (it is already a scalar element type).
static const Record *getElementTypeRecord(const Record *R) {
  if (R->isSubClassOf("VectorType"))
    return R->getValueAsDef("ElementType");
  assert(R->isSubClassOf("HLSLType"));
  return R;
}

//===----------------------------------------------------------------------===//
// Type information
//===----------------------------------------------------------------------===//

namespace {

/// Classifies how a type varies across overloads.
enum TypeKindEnum {
  TK_Varying = 0,      ///< Type matches the full varying type (e.g. float3).
  TK_ElemType = 1,     ///< Type is the scalar element type (e.g. float).
  TK_VaryingShape = 2, ///< Type uses the varying shape with a fixed element.
  TK_FixedType = 3,    ///< Type is a fixed concrete type (e.g. "half2").
  TK_Void = 4          ///< Type is void (only valid for return types).
};

/// Metadata describing how a type (argument or return) varies across overloads.
struct TypeInfo {
  /// Classification of how this type varies across overloads.
  TypeKindEnum Kind = TK_Varying;

  /// Fixed type name (e.g. "half2") for types with a concrete type that does
  /// not vary across overloads. Empty for varying types.
  std::string FixedType;

  /// Element type name for TK_VaryingShape types (e.g. "bool" for
  /// VaryingShape<BoolTy>). Empty for other type kinds.
  StringRef ShapeElemType;

  /// Explicit parameter name (e.g. "eta"). Empty to use the default "p0",
  /// "p1", ... naming. Only meaningful for argument types.
  StringRef Name;

  /// Construct a TypeInfo from a TableGen record.
  static TypeInfo resolve(const Record *Rec) {
    TypeInfo TI;
    if (Rec->getName() == "VoidTy") {
      TI.Kind = TK_Void;
    } else if (Rec->getName() == "Varying") {
      TI.Kind = TK_Varying;
    } else if (Rec->getName() == "VaryingElemType") {
      TI.Kind = TK_ElemType;
    } else if (Rec->isSubClassOf("VaryingShape")) {
      TI.Kind = TK_VaryingShape;
      TI.ShapeElemType =
          Rec->getValueAsDef("ElementType")->getValueAsString("Name");
    } else if (Rec->isSubClassOf("VectorType") ||
               Rec->isSubClassOf("HLSLType")) {
      TI.Kind = TK_FixedType;
      TI.FixedType = getFixedTypeName(Rec);
    } else {
      llvm_unreachable("unhandled record for type resolution");
    }
    return TI;
  }

  /// Resolve this type to a concrete type name string.
  /// \p ElemType is the scalar element type for the current overload.
  /// \p FormatVarying formats a scalar element type into the shaped type name.
  std::string
  toTypeString(StringRef ElemType,
               function_ref<std::string(StringRef)> FormatVarying) const {
    switch (Kind) {
    case TK_Void:
      return "void";
    case TK_Varying:
      return FormatVarying(ElemType);
    case TK_ElemType:
      return ElemType.str();
    case TK_VaryingShape:
      return FormatVarying(ShapeElemType);
    case TK_FixedType:
      assert(!FixedType.empty() && "TK_FixedType requires non-empty FixedType");
      return FixedType;
    }
    llvm_unreachable("unhandled TypeKindEnum");
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Availability helpers
//===----------------------------------------------------------------------===//

static void emitAvailability(raw_ostream &OS, StringRef Version,
                             bool Use16Bit = false) {
  if (Use16Bit)
    OS << "_HLSL_16BIT_AVAILABILITY(shadermodel, " << Version << ")\n";
  else
    OS << "_HLSL_AVAILABILITY(shadermodel, " << Version << ")\n";
}

static std::string getVersionString(const Record *SM) {
  unsigned Major = SM->getValueAsInt("Major");
  unsigned Minor = SM->getValueAsInt("Minor");
  if (Major == 0 && Minor == 0)
    return "";
  return (Twine(Major) + "." + Twine(Minor)).str();
}

//===----------------------------------------------------------------------===//
// Type work item — describes one element type to emit overloads for
//===----------------------------------------------------------------------===//

namespace {

/// A single entry in the worklist of types to process for an intrinsic.
struct TypeWorkItem {
  /// Element type name (e.g. "half", "float"). Empty for fixed-arg-only
  /// intrinsics with no type expansion.
  StringRef ElemType;

  /// Version string for the availability attribute (e.g. "6.2"). Empty if
  /// no availability annotation is needed.
  std::string Availability;

  /// If true, emit _HLSL_16BIT_AVAILABILITY instead of _HLSL_AVAILABILITY.
  bool Use16BitAvail = false;

  /// If true, wrap overloads in #ifdef __HLSL_ENABLE_16_BIT / #endif.
  bool NeedsIfdefGuard = false;
};

} // anonymous namespace

/// Fixed canonical ordering for overload types. Types are grouped as:
///   0: conditionally-16-bit (half)
///   1-2: 16-bit integers (int16_t, uint16_t) — ifdef-guarded
///   3+: regular types (bool, int, uint, int64_t, uint64_t, float, double)
/// Within each group, signed precedes unsigned, smaller precedes larger,
/// and integer types precede floating-point types.
static int getTypeSortPriority(const Record *ET) {
  return StringSwitch<int>(ET->getValueAsString("Name"))
      .Case("half", 0)
      .Case("int16_t", 1)
      .Case("uint16_t", 2)
      .Case("bool", 3)
      .Case("int", 4)
      .Case("uint", 5)
      .Case("int64_t", 7)
      .Case("uint64_t", 8)
      .Case("float", 9)
      .Case("double", 10)
      .Default(11);
}

//===----------------------------------------------------------------------===//
// Overload context — shared state across all overloads of one intrinsic
//===----------------------------------------------------------------------===//

namespace {

/// Shared state for emitting all overloads of a single HLSL intrinsic.
struct OverloadContext {
  /// Output stream to write generated code to.
  raw_ostream &OS;

  /// Builtin name for _HLSL_BUILTIN_ALIAS (e.g. "__builtin_hlsl_dot").
  /// Empty for inline/detail intrinsics.
  StringRef Builtin;

  /// __detail helper function to call (e.g. "refract_impl").
  /// Empty for alias and inline-body intrinsics.
  StringRef DetailFunc;

  /// Literal inline function body (e.g. "return p0;").
  /// Empty for alias and detail intrinsics.
  StringRef Body;

  /// The HLSL function name to emit (e.g. "dot", "refract").
  StringRef FuncName;

  /// Metadata describing the return type and its variation behavior.
  TypeInfo RetType;

  /// Per-argument metadata describing type and variation behavior.
  SmallVector<TypeInfo, 4> Args;

  /// Whether to emit the function as constexpr.
  bool IsConstexpr = false;

  /// Whether to emit the __attribute__((convergent)) annotation.
  bool IsConvergent = false;

  /// Whether any fixed arg has a 16-bit integer type (e.g. int16_t).
  bool Uses16BitType = false;

  /// Whether any fixed arg has a conditionally-16-bit type (half).
  bool UsesConditionally16BitType = false;

  explicit OverloadContext(raw_ostream &OS) : OS(OS) {}
};

} // anonymous namespace

/// Emit a complete function declaration or definition with pre-resolved types.
static void emitDeclaration(const OverloadContext &Ctx, StringRef RetType,
                            ArrayRef<std::string> ArgTypes) {
  raw_ostream &OS = Ctx.OS;
  bool IsDetail = !Ctx.DetailFunc.empty();
  bool IsInline = !Ctx.Body.empty();
  bool HasBody = IsDetail || IsInline;

  bool EmitNames = HasBody || llvm::any_of(Ctx.Args, [](const TypeInfo &A) {
                     return !A.Name.empty();
                   });

  auto GetParamName = [&](unsigned I) -> std::string {
    if (!Ctx.Args[I].Name.empty())
      return Ctx.Args[I].Name.str();
    return ("p" + Twine(I)).str();
  };

  if (!HasBody)
    OS << "_HLSL_BUILTIN_ALIAS(" << Ctx.Builtin << ")\n";
  if (Ctx.IsConvergent)
    OS << "__attribute__((convergent)) ";
  if (HasBody)
    OS << (Ctx.IsConstexpr ? "constexpr " : "inline ");
  OS << RetType << " " << Ctx.FuncName << "(";

  {
    ListSeparator LS;
    for (unsigned I = 0, N = ArgTypes.size(); I < N; ++I) {
      OS << LS << ArgTypes[I];
      if (EmitNames)
        OS << " " << GetParamName(I);
    }
  }

  if (IsDetail) {
    OS << ") {\n  return __detail::" << Ctx.DetailFunc << "(";
    ListSeparator LS;
    for (unsigned I = 0, N = ArgTypes.size(); I < N; ++I)
      OS << LS << GetParamName(I);
    OS << ");\n}\n";
  } else if (IsInline) {
    OS << ") { " << Ctx.Body << " }\n";
  } else {
    OS << ");\n";
  }
}

/// Emit a single overload declaration by resolving all types through
/// \p FormatVarying, which maps element types to their shaped form.
static void emitOverload(const OverloadContext &Ctx, StringRef ElemType,
                         function_ref<std::string(StringRef)> FormatVarying) {
  std::string RetType = Ctx.RetType.toTypeString(ElemType, FormatVarying);
  SmallVector<std::string> ArgTypes;
  for (const TypeInfo &TI : Ctx.Args)
    ArgTypes.push_back(TI.toTypeString(ElemType, FormatVarying));
  emitDeclaration(Ctx, RetType, ArgTypes);
}

/// Emit a scalar overload for the given element type.
static void emitScalarOverload(const OverloadContext &Ctx, StringRef ElemType) {
  emitOverload(Ctx, ElemType, [](StringRef ET) { return ET.str(); });
}

/// Emit a vector overload for the given element type and vector size.
static void emitVectorOverload(const OverloadContext &Ctx, StringRef ElemType,
                               unsigned VecSize) {
  emitOverload(Ctx, ElemType, [VecSize](StringRef ET) {
    return getVectorTypeName(ET, VecSize);
  });
}

/// Emit a matrix overload for the given element type and matrix dimensions.
static void emitMatrixOverload(const OverloadContext &Ctx, StringRef ElemType,
                               unsigned Rows, unsigned Cols) {
  emitOverload(Ctx, ElemType, [Rows, Cols](StringRef ET) {
    return getMatrixTypeName(ET, Rows, Cols);
  });
}

//===----------------------------------------------------------------------===//
// Main emission logic
//===----------------------------------------------------------------------===//

/// Build an OverloadContext from an HLSLBuiltin record.
static void buildOverloadContext(const Record *R, OverloadContext &Ctx) {
  Ctx.Builtin = R->getValueAsString("Builtin");
  Ctx.DetailFunc = R->getValueAsString("DetailFunc");
  Ctx.Body = R->getValueAsString("Body");
  Ctx.FuncName = R->getValueAsString("Name");
  Ctx.IsConstexpr = R->getValueAsBit("IsConstexpr");
  Ctx.IsConvergent = R->getValueAsBit("IsConvergent");

  // Note use of 16-bit fixed types in the overload context.
  auto Update16BitFlags = [&Ctx](const Record *Rec) {
    const Record *ElemTy = getElementTypeRecord(Rec);
    Ctx.Uses16BitType |= ElemTy->getValueAsBit("Is16Bit");
    Ctx.UsesConditionally16BitType |=
        ElemTy->getValueAsBit("IsConditionally16Bit");
  };

  // Resolve return and argument types.
  const Record *RetRec = R->getValueAsDef("ReturnType");
  Ctx.RetType = TypeInfo::resolve(RetRec);
  if (Ctx.RetType.Kind == TK_FixedType)
    Update16BitFlags(RetRec);

  std::vector<const Record *> ArgRecords = R->getValueAsListOfDefs("Args");
  std::vector<StringRef> ParamNames = R->getValueAsListOfStrings("ParamNames");

  for (const auto &[I, Arg] : llvm::enumerate(ArgRecords)) {
    TypeInfo TI = TypeInfo::resolve(Arg);
    if (I < ParamNames.size())
      TI.Name = ParamNames[I];
    if (TI.Kind == TK_FixedType)
      Update16BitFlags(Arg);
    Ctx.Args.push_back(TI);
  }
}

/// Build the worklist of element types to emit overloads for, sorted in
/// canonical order (see getTypeSortPriority).
static void buildWorklist(const Record *R,
                          SmallVectorImpl<TypeWorkItem> &Worklist,
                          const OverloadContext &Ctx) {
  const Record *AvailRec = R->getValueAsDef("Availability");
  std::string Availability = getVersionString(AvailRec);
  bool AvailabilityIsAtLeastSM6_2 = AvailRec->getValueAsInt("Major") > 6 ||
                                    (AvailRec->getValueAsInt("Major") == 6 &&
                                     AvailRec->getValueAsInt("Minor") >= 2);

  std::vector<const Record *> VaryingTypeRecords =
      R->getValueAsListOfDefs("VaryingTypes");

  // Populate the availability and guard fields of a TypeWorkItem based on
  // whether the type is 16-bit, conditionally 16-bit, or a regular type.
  auto SetAvailability = [&](TypeWorkItem &Item, bool Is16Bit,
                             bool IsCond16Bit) {
    Item.NeedsIfdefGuard = Is16Bit;
    if (Is16Bit || IsCond16Bit) {
      if (AvailabilityIsAtLeastSM6_2) {
        Item.Availability = Availability;
      } else {
        Item.Availability = SM6_2;
        Item.Use16BitAvail = IsCond16Bit;

        // Note: If Availability = x where x < 6.2 and a half type is used,
        // neither _HLSL_AVAILABILITY(shadermodel, x) nor
        // _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2) are correct:
        //
        //   _HLSL_AVAILABILITY(shadermodel, x) will set the availbility for the
        //   half overload to x even when 16-bit types are enabled, but x < 6.2
        //   and 6.2 is required for 16-bit half.
        //
        //   _HLSL_16BIT_AVAILABILITY(shadermodel, 6.2) will set the
        //   availability for the half overload to 6.2 when 16-bit types are
        //   enabled, but there will be no availability set when 16-bit types
        //   are not enabled.
        //
        // A possible solution to this is to make _HLSL_16BIT_AVAILABILITY
        // accept 3 args: (shadermodel, X, Y) where X is the availability for
        // the 16-bit half type overload (which will typically be 6.2), and Y is
        // the availability for the non-16-bit half overload. However, this
        // situation does not currently arise, so we just assert below that this
        // case will never occur.
        assert(
            !(IsCond16Bit && !Availability.empty()) &&
            "Can not handle availability for an intrinsic using half types and"
            " which has an explicit shader model requirement older than 6.2");
      }
    } else {
      Item.Availability = Availability;
    }
  };

  // If no Varying types are specified, just add a single work item.
  // This is for HLSLBuiltin records that don't use Varying types.
  if (VaryingTypeRecords.empty()) {
    TypeWorkItem Item;
    SetAvailability(Item, Ctx.Uses16BitType, Ctx.UsesConditionally16BitType);
    Worklist.push_back(Item);
    return;
  }

  // Sort Varying types so that overloads are always emitted in canonical order.
  llvm::sort(VaryingTypeRecords, [](const Record *A, const Record *B) {
    return getTypeSortPriority(A) < getTypeSortPriority(B);
  });

  // Add a work item for each Varying element type.
  for (const Record *ElemTy : VaryingTypeRecords) {
    TypeWorkItem Item;
    Item.ElemType = ElemTy->getValueAsString("Name");
    bool Is16Bit = Ctx.Uses16BitType || ElemTy->getValueAsBit("Is16Bit");
    bool IsCond16Bit = Ctx.UsesConditionally16BitType ||
                       ElemTy->getValueAsBit("IsConditionally16Bit");
    SetAvailability(Item, Is16Bit, IsCond16Bit);
    Worklist.push_back(Item);
  }
}

/// Emit a Doxygen documentation comment from the Doc field.
static void emitDocComment(raw_ostream &OS, const Record *R) {
  StringRef Doc = R->getValueAsString("Doc");
  if (Doc.empty())
    return;
  Doc = Doc.trim();
  SmallVector<StringRef> DocLines;
  Doc.split(DocLines, '\n');
  for (StringRef Line : DocLines) {
    if (Line.empty())
      OS << "///\n";
    else
      OS << "/// " << Line << "\n";
  }
}

/// Process the worklist: emit all shape variants for each type with
/// availability annotations and #ifdef guards.
static void emitWorklistOverloads(raw_ostream &OS, const OverloadContext &Ctx,
                                  ArrayRef<TypeWorkItem> Worklist,
                                  bool EmitScalarOverload,
                                  ArrayRef<int64_t> VectorSizes,
                                  ArrayRef<const Record *> MatrixDimensions) {
  bool InIfdef = false;
  for (const TypeWorkItem &Item : Worklist) {
    if (Item.NeedsIfdefGuard && !InIfdef) {
      OS << "#ifdef __HLSL_ENABLE_16_BIT\n";
      InIfdef = true;
    }

    auto EmitAvail = [&]() {
      if (!Item.Availability.empty())
        emitAvailability(OS, Item.Availability, Item.Use16BitAvail);
    };

    if (EmitScalarOverload) {
      EmitAvail();
      emitScalarOverload(Ctx, Item.ElemType);
    }
    for (int64_t N : VectorSizes) {
      EmitAvail();
      emitVectorOverload(Ctx, Item.ElemType, N);
    }
    for (const Record *MD : MatrixDimensions) {
      EmitAvail();
      emitMatrixOverload(Ctx, Item.ElemType, MD->getValueAsInt("Rows"),
                         MD->getValueAsInt("Cols"));
    }

    if (InIfdef) {
      bool NextIsUnguarded =
          (&Item == &Worklist.back()) || !(&Item + 1)->NeedsIfdefGuard;
      if (NextIsUnguarded) {
        OS << "#endif\n";
        InIfdef = false;
      }
    }

    OS << "\n";
  }
}

/// Emit all overloads for a single HLSLBuiltin record.
static void emitBuiltinOverloads(raw_ostream &OS, const Record *R) {
  OverloadContext Ctx(OS);
  buildOverloadContext(R, Ctx);

  SmallVector<TypeWorkItem> Worklist;
  buildWorklist(R, Worklist, Ctx);

  emitDocComment(OS, R);
  OS << "// " << Ctx.FuncName << " overloads\n";

  // Emit a scalar overload if a scalar Varying overload was requested.
  // If no Varying types are used at all, emit a scalar overload to handle
  // emitting a single overload for fixed-typed args or arg-less functions.
  bool EmitScalarOverload = R->getValueAsBit("VaryingScalar") ||
                            R->getValueAsListOfDefs("VaryingTypes").empty();

  std::vector<int64_t> VectorSizes = R->getValueAsListOfInts("VaryingVecSizes");
  std::vector<const Record *> MatrixDimensions =
      R->getValueAsListOfDefs("VaryingMatDims");

  // Sort vector sizes and matrix dimensions for consistent output order.
  llvm::sort(VectorSizes);
  llvm::sort(MatrixDimensions, [](const Record *A, const Record *B) {
    int RowA = A->getValueAsInt("Rows"), RowB = B->getValueAsInt("Rows");
    if (RowA != RowB)
      return RowA < RowB;
    return A->getValueAsInt("Cols") < B->getValueAsInt("Cols");
  });

  emitWorklistOverloads(OS, Ctx, Worklist, EmitScalarOverload, VectorSizes,
                        MatrixDimensions);
}

/// Emit alias overloads for a single HLSLBuiltin record.
/// Skips records that have inline bodies (DetailFunc or Body).
static void emitAliasBuiltin(raw_ostream &OS, const Record *R) {
  if (!R->getValueAsString("DetailFunc").empty() ||
      !R->getValueAsString("Body").empty())
    return;
  emitBuiltinOverloads(OS, R);
}

/// Emit inline overloads for a single HLSLBuiltin record.
/// Skips records that are pure alias declarations.
static void emitInlineBuiltin(raw_ostream &OS, const Record *R) {
  if (R->getValueAsString("DetailFunc").empty() &&
      R->getValueAsString("Body").empty())
    return;
  emitBuiltinOverloads(OS, R);
}

void clang::EmitHLSLAliasIntrinsics(const RecordKeeper &Records,
                                    raw_ostream &OS) {
  OS << "// This file is auto-generated by clang-tblgen from "
        "HLSLIntrinsics.td.\n";
  OS << "// Do not edit this file directly.\n\n";

  for (const Record *R : Records.getAllDerivedDefinitions("HLSLBuiltin"))
    emitAliasBuiltin(OS, R);
}

void clang::EmitHLSLInlineIntrinsics(const RecordKeeper &Records,
                                     raw_ostream &OS) {
  OS << "// This file is auto-generated by clang-tblgen from "
        "HLSLIntrinsics.td.\n";
  OS << "// Do not edit this file directly.\n\n";

  for (const Record *R : Records.getAllDerivedDefinitions("HLSLBuiltin"))
    emitInlineBuiltin(OS, R);
}
