//===- TypedMemoryInference.cpp - ASTContext TMO support. -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TMO components of the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypedMemoryTypeDescriptor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/STLExtras.h"
#include <variant>

using namespace clang;
namespace { // enum->string conversion namespace
template <class T> struct EnumStringMap;

template <> struct EnumStringMap<TypedMemoryLayoutSemantics> {
  static constexpr std::pair<TypedMemoryLayoutSemantics, const char *> Map[] = {
      {TypedMemoryLayoutSemantics::DataPointer, "DataPointer"},
      {TypedMemoryLayoutSemantics::StructPointer, "StructPointer"},
      {TypedMemoryLayoutSemantics::ImmutablePointer, "ImmutablePointer"},
      {TypedMemoryLayoutSemantics::AnonymousPointer, "AnonymousPointer"},
      {TypedMemoryLayoutSemantics::ReferenceCount, "ReferenceCount"},
      {TypedMemoryLayoutSemantics::ResourceHandle, "ResourceHandle"},
      {TypedMemoryLayoutSemantics::SpatialBounds, "SpatialBounds"},
      {TypedMemoryLayoutSemantics::TaintedData, "TaintedData"},
      {TypedMemoryLayoutSemantics::GenericData, "GenericData"}};
};

template <> struct EnumStringMap<TypedMemoryTypeFlags> {
  static constexpr std::pair<TypedMemoryTypeFlags, const char *> Map[] = {
      {TypedMemoryTypeFlags::IsPolymorphic, "IsPolymorphic"},
      {TypedMemoryTypeFlags::HasMixedUnions, "HasMixedUnions"}};
};

template <> struct EnumStringMap<TypedMemoryTypeKind> {
  static constexpr std::pair<TypedMemoryTypeKind, const char *> Map[] = {
      {TypedMemoryTypeKind::KindC, "KindC"},
      {TypedMemoryTypeKind::KindObjectiveC, "KindObjectiveC"},
      {TypedMemoryTypeKind::KindSwift, "KindSwift"},
      {TypedMemoryTypeKind::KindCxx, "KindCxx"}};
};

template <> struct EnumStringMap<TypedMemoryCallsiteFlags> {
  static constexpr std::pair<TypedMemoryCallsiteFlags, const char *> Map[] = {
      {TypedMemoryCallsiteFlags::FixedSize, "FixedSize"},
      {TypedMemoryCallsiteFlags::Array, "Array"},
      {TypedMemoryCallsiteFlags::HeaderPrefixedArray, "HeaderPrefixedArray"}};
};

template <class EnumType, class F>
static std::enable_if_t<llvm::is_bitmask_enum<EnumType>::value, void>
MapBitflag(EnumType Flags, F Callback) {
  for (auto [Flag, Description] : EnumStringMap<EnumType>::Map) {
    if ((Flags & Flag) != Flag)
      continue;
    Callback(Flag, Description);
  }
}

template <class EnumType, class F>
static std::enable_if_t<!llvm::is_bitmask_enum<EnumType>::value, void>
MapBitflag(EnumType Flags, F Callback) {
  for (auto [Flag, Description] : EnumStringMap<EnumType>::Map) {
    if (Flags != Flag)
      continue;
    Callback(Flag, Description);
  }
}
} // namespace

void InferredAllocationType::Object::print(
    llvm::raw_ostream &OS, llvm::function_ref<void(QualType)> QuoteType) const {
  QuoteType(Type);
}

void InferredAllocationType::Array::print(
    llvm::raw_ostream &OS, llvm::function_ref<void(QualType)> QuoteType) const {
  QuoteType(ElementType);
}

void InferredAllocationType::PrefixedArray::print(
    llvm::raw_ostream &OS, llvm::function_ref<void(QualType)> QuoteType) const {
  OS << "header prefixed array of {";
  QuoteType(HeaderType);
  OS << ':';
  if (ElementType)
    QuoteType(*ElementType);
  else
    OS << "<unknown element type>";
  OS << '}';
}

void InferredAllocationType::Tuple::print(
    llvm::raw_ostream &OS, llvm::function_ref<void(QualType)> QuoteType) const {
  OS << "tuple of (";
  llvm::interleaveComma(Entries, OS, QuoteType);
  OS << ')';
}

void InferredAllocationType::UnknownLayout::print(
    llvm::raw_ostream &OS, llvm::function_ref<void(QualType)> QuoteType) const {
  OS << "indeterminate set of {";
  llvm::interleaveComma(Entries, OS, QuoteType);
  OS << '}';
}

void InferredAllocationType::print(llvm::raw_ostream &OS,
                                   const PrintingPolicy &Policy) const {
  // This reimplements the diagnostic engine's type printing logic to minimize
  // changes in test output.
  auto QuoteType = [&](QualType Type) {
    QualType UnqualifiedType = Type.getUnqualifiedType();
    QualType DesugaredType(Type->getUnqualifiedDesugaredType(), 0);
    std::string UnqualifiedTypeName;
    llvm::raw_string_ostream UnqualifiedTypeStream(UnqualifiedTypeName);
    UnqualifiedType.print(UnqualifiedTypeStream, Policy);
    OS << "'" << UnqualifiedTypeName << "'";
    if (const auto *VTy = UnqualifiedType->getAs<VectorType>()) {
      const char *Values = VTy->getNumElements() > 1 ? "values" : "value";
      OS << " (vector of " << VTy->getNumElements() << " '"
         << VTy->getElementType().getAsString(Policy) << "' " << Values << ")";
      return;
    }
    std::string DesugaredTypeName;
    llvm::raw_string_ostream DesugaredTypeNameStream(DesugaredTypeName);
    DesugaredType.print(DesugaredTypeNameStream, Policy);
    if (DesugaredTypeName != UnqualifiedTypeName)
      OS << " (aka '" << DesugaredTypeName << "')";
  };
  visit([&](const auto &V) { V.print(OS, QuoteType); });
}

std::string InferredAllocationType::describe(const ASTContext &Ctx) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  print(OS, Ctx.getPrintingPolicy());
  return Result;
}

TypedMemoryInference &ASTContext::getTMOInference() const {
  if (!TypeInference)
    TypeInference = std::make_unique<TypedMemoryInference>();
  return *TypeInference;
}

StringRef
TypedMemoryInference::getTypeSummaryDescription(TypedMemorySummary Summary) {
  auto [Entry, Inserted] = TypeSummaryDescriptionCache.insert(
      std::make_pair(Summary.value(), std::unique_ptr<std::string>{}));
  if (!Inserted)
    return *Entry->second;
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  auto LogFlags = [&OS](const char *Name, auto Flags) {
    OS << '"' << Name << '"' << ": ";
    constexpr bool IsBitfield = llvm::is_bitmask_enum<decltype(Flags)>::value;
    if (IsBitfield)
      OS << "[ ";
    bool HasEmitted = false;
    MapBitflag(Flags, [&OS, &HasEmitted](decltype(Flags) Flag,
                                         const char *Description) {
      OS << (HasEmitted ? ", " : "") << '"' << Description << '"';
      HasEmitted = true;
    });
    if (IsBitfield)
      OS << (HasEmitted ? " ]" : "]");
  };
  LogFlags("LayoutSemantics", Summary.LayoutSemantics);
  OS << ", ";
  LogFlags("TypeFlags", Summary.TypeFlags);
  OS << ", ";
  LogFlags("TypeKind", Summary.TypeKind);
  OS << ", ";
  LogFlags("CallsiteFlags", Summary.CallsiteFlags);
  Entry->second = std::make_unique<std::string>(std::move(Str));
  return *(Entry->second);
}

TypedMemoryInference::TypedMemoryDescriptor
TypedMemoryInference::getDescriptor(const ASTContext &Ctx, QualType QT,
                                    OverloadedOperatorKind OverloadedOperator,
                                    TypedMemoryCallsiteFlags Flags) {
  if (QT->isDependentType())
    return TypedMemoryDescriptor{0, {}, ""};

  QT = QT->getCanonicalTypeUnqualified();
  bool IsCXXOperator = false;
  switch (OverloadedOperator) {
  case OO_Array_New:
  case OO_Array_Delete:
    Flags |= TypedMemoryCallsiteFlags::Array;
    LLVM_FALLTHROUGH;
  case OO_New:
  case OO_Delete:
    IsCXXOperator = true;
    break;
  case OO_None:
    break;
  default:
    llvm_unreachable("Invalid allocation operator.");
  }

  TypedMemoryTypeKind TypeKind =
      IsCXXOperator ? TypedMemoryTypeKind::KindCxx : TypedMemoryTypeKind::KindC;

  TypeDescriptorMapKeyTy Key = {QT.getTypePtr(), Flags, TypeKind};
  auto ViewForEntry = [](const TypedMemoryDescriptorMapValueTy &Entry) {
    return TypedMemoryDescriptor{Entry.IdentityHash, Entry.Summary,
                                 Entry.TypeDescription ? *Entry.TypeDescription
                                                       : StringRef()};
  };
  auto InsertResult =
      TypeDescriptorMap.insert({Key, TypedMemoryDescriptorMapValueTy{}});
  auto Found = InsertResult.first;
  bool Inserted = InsertResult.second;
  if (!Inserted) {
    if (!Found->second.IsIncomplete)
      return ViewForEntry(Found->second);
  }

  auto FinalizeDescriptor = [this, &Ctx, &ViewForEntry, &Found,
                             QT](TypedMemorySummary Summary,
                                 uint32_t IdentityHash) {
    TypedMemoryDescriptorMapValueTy Descriptor;
    Descriptor.IsIncomplete = QT->isIncompleteType();
    Descriptor.Summary = Summary;
    Descriptor.IdentityHash = IdentityHash;

    if (!Ctx.getDiagnostics().isIgnored(diag::remark_tmo_passed_type,
                                        SourceLocation())) {
      std::string Description;
      llvm::raw_string_ostream OS(Description);
      OS << "{ \"Summary\": { " << getTypeSummaryDescription(Descriptor.Summary)
         << " }, \"TypeHash\": " << Descriptor.IdentityHash
         << (QT->isIncompleteType() ? ", \"IncompleteType\": true" : "")
         << " }";
      Descriptor.TypeDescription = std::make_unique<std::string>(Description);
    }
    return ViewForEntry(Found->second = std::move(Descriptor));
  };

  TypedMemorySummary Summary;
  Summary.Version = 0;
  Summary.CallsiteFlags = Flags;
  Summary.TypeKind = TypeKind;

  if (QT->isIncompleteType())
    return FinalizeDescriptor(
        Summary, TypedMemoryTypeDescriptor::computeTypeNameHash(QT));

  // Avoid pathological compile times on types larger than 4MB.
  static const uint64_t MaximumTypeSizeInBits = 1U << 25;
  if (Ctx.getTypeSize(QT) >= MaximumTypeSizeInBits)
    return FinalizeDescriptor(
        {}, TypedMemoryTypeDescriptor::computeTypeNameHash(QT));

  // Build the flattened memory layout
  TypedMemoryLayoutBuilder Builder(Ctx, QT);
  Builder.build();

  // Create the type descriptor
  TypedMemoryTypeDescriptor TypeDescriptor;
  TypeDescriptor.initialize(QT, Builder.getLayout());
  Summary.LayoutSemantics = TypeDescriptor.getCoalescedLayoutSemantics();
  Summary.TypeFlags = TypeDescriptor.getTypeFlags();
  return FinalizeDescriptor(Summary, TypeDescriptor.computeTypeHash());
}

void TypedMemoryInference::setInferredInfoForCall(const CallExpr *Call,
                                                  InferredTypeInfo Info) {
  auto [InsertionPoint, Inserted] = InferredTMOCallTypes.insert({Call, Info});
  // We are keyed on the Call expression itself, so shouldn't be recording this
  // multiple times.
  assert(Inserted);
}

std::optional<InferredTypeInfo>
TypedMemoryInference::getInferredInfoForCall(const ASTContext &Ctx,
                                             const CallExpr *Call) const {
  if (!Ctx.getLangOpts().TypedMemoryOperations)
    return std::nullopt;
  if (Call->getDependence() != ExprDependence::None)
    return std::nullopt;

  if (auto Found = InferredTMOCallTypes.find(Call);
      Found != InferredTMOCallTypes.end())
    return Found->second;

#ifndef NDEBUG
  // If we get here something has gone wrong: either we've failed to perform
  // inference on a call that we should have, or the caller has incorrectly
  // called us on an inappropriate function. Either case is wrong so we assert
  // in debug/relassert builds and warn in release.
  //
  // We're currently going to be forgiving as we don't want to break builds
  // with error diagnostics until we're sure this does not actually fire in
  // practice.
  DiagnosticsEngine &Diags = Ctx.getDiagnostics();
  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee || !Callee->getAttr<TypedMemoryAttr>()) {
    unsigned DiagID = Diags.getCustomDiagID(
        DiagnosticsEngine::Warning,
        "Requesting TMO inference result for non-TMO applicable call %0");
    Diags.Report(Call->getBeginLoc(), DiagID) << Call << Call->getSourceRange();
    return std::nullopt;
  }

  unsigned DiagID =
      Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                            "Requesting inference result for %0 but not found");
  Diags.Report(Call->getBeginLoc(), DiagID) << Call << Call->getSourceRange();
#endif

  return std::nullopt;
}

// Strip the trivial case of a variable reference: constant inited variables.
//
// TODO: Traverse linear assignment
//   int_type_t sz = sizeof(X);
//   sz += sizeof(Y);
//   malloc(sz);
//
// this is seen a lot in real world, and we don't detect the types involved,
// let alone accurately inferring the actual allocation type.
static const Expr *stripDeclRef(const ASTContext &Ctx, const DeclRefExpr *DRE) {
  const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return DRE;
  QualType DeclType = VD->getType();
  if (!DeclType->isIntegerType() || !DeclType.isConstant(Ctx))
    return DRE;
  if (const Expr *Init = VD->getInit())
    return Init;
  return DRE;
}

// Handle multiplication (order independent):
//   non-type * non-type => non-type
//   type * non-type => array-type
//   array-type * non-type => array-type
//   type * type => indeterminate-type
//   indeterminate-type * any => indeterminate-type
static std::optional<InferredAllocationType>
mergeMultiplies(std::optional<InferredAllocationType> LHS,
                std::optional<InferredAllocationType> RHS) {
  if (!LHS && !RHS)
    return std::nullopt;

  // If both sides are valid we have some variation of sizeof(A) * sizeof(B)
  // which we can't really reason about
  if (LHS && RHS) {
    SmallVector<QualType, 2> Entries;
    LHS->extractRelatedTypes(Entries);
    RHS->extractRelatedTypes(Entries);
    bool IsConstantSize = LHS->isConstantSize() && RHS->isConstantSize();
    return InferredAllocationType::createUnknownLayout(std::move(Entries),
                                                       IsConstantSize);
  }

  const InferredAllocationType &Type = LHS ? *LHS : *RHS;
  // Already an array, so we've just got something like sizeof(A) * x * y
  if (Type.isArray())
    return Type;

  // This has the effect of flattening (sizeof(A) + sizeof(B)) * x
  // but for now that's ok
  if (auto ElementType = Type.primaryType())
    return InferredAllocationType::createArray(*ElementType,
                                               /*IsConstantSize=*/false);
  return std::nullopt;
}

// Handle addition all order independent, except tuples
//   prefixed-array + indeterminate => prefixed-array
//   prefixed-array + non-type => prefixed-array
//   prefixed-array + prefixed-array => indeterminate
//   array-type + array-type:
//     if element type match
//   type + (non-type or indeterminate):
//     if type has a flexible array member => prefixed-array
//     else => indeterminate
//   type + array-type => prefixed-array
//   type + type + ... => tuple-type (type, type, ...)
static std::optional<InferredAllocationType>
mergeAddition(const ASTContext &Ctx, std::optional<InferredAllocationType> LHS,
              std::optional<InferredAllocationType> RHS) {
  if (!LHS && !RHS)
    return std::nullopt;

  bool IsConstantSize =
      LHS && RHS && LHS->isConstantSize() && RHS->isConstantSize();

  auto ConstructUnknownLayout = [&]() {
    llvm::SmallVector<QualType, 2> RelatedTypes;
    if (LHS)
      LHS->extractRelatedTypes(RelatedTypes);
    if (RHS)
      RHS->extractRelatedTypes(RelatedTypes);
    return InferredAllocationType::createUnknownLayout(std::move(RelatedTypes),
                                                       IsConstantSize);
  };

  // type-with-flexible-array + unknown => prefixed-array; else
  // any + unknown => unknown
  if (!LHS || !RHS || LHS->isUnknownLayout() || RHS->isUnknownLayout()) {
    auto *Type = LHS ? LHS->asObject() : nullptr;
    if (!Type)
      Type = RHS ? RHS->asObject() : nullptr;
    if (!Type)
      return ConstructUnknownLayout();
    std::optional<QualType> ElementType =
        Ctx.findFlexibleArrayElementType(Type->Type);
    if (!ElementType)
      return ConstructUnknownLayout();
    return InferredAllocationType::createPrefixedArray(Type->Type, *ElementType,
                                                       IsConstantSize);
  }

  // Both LHS and RHS are present and have known layout from here on.
  assert(LHS && RHS && !LHS->isUnknownLayout() && !RHS->isUnknownLayout());

  // array<T> + array<T> => array<T>
  // array<T> + array<U> => unknown
  if (LHS->isArray() && RHS->isArray()) {
    const auto &LeftType = *LHS->asArray();
    const auto &RightType = *RHS->asArray();
    if (!Ctx.hasSameUnqualifiedType(LeftType.ElementType,
                                    RightType.ElementType))
      return ConstructUnknownLayout();
    return InferredAllocationType::createArray(LeftType.ElementType,
                                               IsConstantSize);
  }

  // If both sides are header prefixed arrays, mark it as an indeterminate type
  if (LHS->isPrefixedArray() && RHS->isPrefixedArray())
    return ConstructUnknownLayout();

  // Somewhat convoluted, but we need to handle
  //   sizeof(Header) + sizeof(Element) * a + sizeof(Element) * b
  auto ExtendPrefixedArray = [&](const InferredAllocationType &Header,
                                 const InferredAllocationType &Tail)
      -> std::optional<InferredAllocationType> {
    if (!Header.isPrefixedArray() || !Tail.isArray())
      return std::nullopt;
    auto *PrefixedArrayType = Header.asPrefixedArray();
    auto *ArrayType = Tail.asArray();
    auto ElementType = PrefixedArrayType->ElementType;
    if (!ElementType) {
      return InferredAllocationType::createPrefixedArray(
          PrefixedArrayType->HeaderType, ArrayType->ElementType,
          IsConstantSize);
    }
    if (!Ctx.hasSameUnqualifiedType(*ElementType, *ArrayType->primaryType()))
      return ConstructUnknownLayout();
    return InferredAllocationType::createPrefixedArray(
        PrefixedArrayType->HeaderType, *ElementType, IsConstantSize);
  };
  if (auto ExtendedPrefixedArray = ExtendPrefixedArray(*LHS, *RHS))
    return ExtendedPrefixedArray;
  if (auto ExtendedPrefixedArray = ExtendPrefixedArray(*RHS, *LHS))
    return ExtendedPrefixedArray;

  // Header prefixed arrays, we just assume any non prefixed-array or non-array
  // type + an array is a prefixed-array
  auto ConstructRealPrefixedArray = [&](const InferredAllocationType &Header,
                                        const InferredAllocationType &Tail)
      -> std::optional<InferredAllocationType> {
    if (Header.isArray() || !Tail.isArray())
      return std::nullopt;
    std::optional<QualType> HeaderType = Header.primaryType();
    std::optional<QualType> ElementType = Tail.primaryType();
    if (!HeaderType || !ElementType)
      return std::nullopt;
    return InferredAllocationType::createPrefixedArray(
        *HeaderType, *ElementType, IsConstantSize);
  };
  if (auto PrefixedArray = ConstructRealPrefixedArray(*LHS, *RHS))
    return PrefixedArray;
  if (auto PrefixedArray = ConstructRealPrefixedArray(*RHS, *LHS))
    return PrefixedArray;

  // Finally deal with the tuple case
  if (!LHS->isArray() && !RHS->isArray()) {
    SmallVector<QualType, 2> Entries;
    LHS->extractRelatedTypes(Entries);
    RHS->extractRelatedTypes(Entries);
    return InferredAllocationType::createTuple(std::move(Entries));
  }
  return ConstructUnknownLayout();
}

namespace {

class AllocationInferenceWalker
    : public ConstStmtVisitor<AllocationInferenceWalker,
                              std::optional<InferredAllocationType>> {
  const ASTContext &Ctx;

public:
  explicit AllocationInferenceWalker(const ASTContext &Ctx) : Ctx(Ctx) {}

  std::optional<InferredAllocationType>
  VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E) {
    if (E->getKind() == UETT_SizeOf)
      return InferredAllocationType::create(E->getTypeOfArgument());
    return std::nullopt;
  }

  std::optional<InferredAllocationType>
  VisitBinaryOperator(const BinaryOperator *E) {
    switch (E->getOpcode()) {
    case BO_Assign:
      return Visit(E->getRHS());
    case BO_Mul:
    case BO_MulAssign:
      return mergeMultiplies(Visit(E->getLHS()), Visit(E->getRHS()));
    case BO_Add:
    case BO_AddAssign:
      return mergeAddition(Ctx, Visit(E->getLHS()), Visit(E->getRHS()));
    case BO_Comma:
      return Visit(E->getRHS());
    default:
      return VisitExpr(E);
    }
  }

  std::optional<InferredAllocationType>
  VisitUnaryOperator(const UnaryOperator *E) {
    switch (E->getOpcode()) {
    case UO_Minus:
    case UO_Plus:
      return Visit(E->getSubExpr());
    default:
      return VisitExpr(E);
    }
  }

  std::optional<InferredAllocationType> VisitDeclRefExpr(const DeclRefExpr *E) {
    const Expr *Stripped = stripDeclRef(Ctx, E);
    if (Stripped != E)
      return Visit(Stripped);
    return std::nullopt;
  }

  std::optional<InferredAllocationType>
  VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
    if (const Expr *Src = E->getSourceExpr())
      return Visit(Src);
    return std::nullopt;
  }

  std::optional<InferredAllocationType>
  VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
    return Visit(E->getResultExpr());
  }

  // The final fallback for unrecognized expressions is to simply traverse all
  // the children and accumulate all nested types.
  // If there is a single non-trivial nested type we propogate it on the
  // assumption that it is sufficiently complex inference to be reasonably
  // accurate.
  // Otherwise we report an UnknownLayout containing the union of all contained
  // types.
  std::optional<InferredAllocationType> VisitExpr(const Expr *E) {
    SmallVector<InferredAllocationType, 2> NestedTypes;
    size_t ChildCount = 0;
    for (const Stmt *SubStmt : E->children()) {
      const Expr *SubExpr = dyn_cast<Expr>(SubStmt);
      if (!SubExpr)
        continue;
      ++ChildCount;
      if (auto InferredType = Visit(SubExpr))
        NestedTypes.push_back(std::move(*InferredType));
    }
    if (NestedTypes.empty())
      return std::nullopt;
    // We don't include single object matches here, as it is very easy for a
    // complex expression to end up only referencing a single visible type,
    // so we're likely overselling our knowledge if we simply propogate that.
    if (NestedTypes.size() == 1 &&
        (ChildCount == 1 || !NestedTypes[0].asObject()))
      return NestedTypes[0];
    llvm::SmallVector<QualType, 2> ReferencedTypes;
    for (auto &Type : NestedTypes)
      Type.extractRelatedTypes(ReferencedTypes);
    return InferredAllocationType::createUnknownLayout(
        std::move(ReferencedTypes),
        /*IsConstantSize=*/false);
  }
};

} // anonymous namespace

static std::optional<InferredAllocationType>
inferAllocationType(const ASTContext &Ctx, const Expr *E) {
  return AllocationInferenceWalker(Ctx).Visit(E);
}

InferredTypeInfo
TypedMemoryInference::inferType(const ASTContext &Ctx, const CallExpr *Call,
                                const Expr &SizeArg,
                                const CastExpr *ContainingCastExpr) {
  assert(Call->getDependence() == ExprDependence::None);
  assert(SizeArg.getDependence() == ExprDependence::None);
  assert(!ContainingCastExpr ||
         ContainingCastExpr->getDependence() == ExprDependence::None);

  auto InsertionResult = InferredTMOCallTypes.insert({Call, {}});
  // Capturing a structured binding is an C++20 extension which we warn about
  auto Found = InsertionResult.first;
  bool Inserted = InsertionResult.second;
  if (!Inserted)
    return Found->second;

  auto RecordResult =
      [&, Found](const Expr *InferredExpr,
                 std::optional<InferredAllocationType> InferredType,
                 TypedMemoryCallsiteFlags Flags) {
        return (Found->second = {InferredType, InferredExpr->IgnoreImplicit(),
                                 Flags});
      };

  QualType CastRecordTarget;
  if (ContainingCastExpr && Call->getCallReturnType(Ctx)->isPointerType()) {
    QualType CastDest = ContainingCastExpr->getType();
    if (CastDest->isAnyPointerType()) {
      CastDest = CastDest->getPointeeType();
      if (const RecordType *CastRecord = CastDest->getAsStructureType())
        CastRecordTarget = QualType(CastRecord, 0);
    }
  }
  auto InferredType = inferAllocationType(Ctx, &SizeArg);
  bool PreferCastType = false;
  if (InferredType && !CastRecordTarget.isNull()) {
    if (auto Inferred = InferredType->primaryType()) {
      PreferCastType = (*Inferred)->isVoidPointerType() ||
                       (*Inferred)->isVoidType() || (*Inferred)->isCharType();
    }
  }
  if (InferredType && !PreferCastType) {
    TypedMemoryCallsiteFlags Flags = TypedMemoryCallsiteFlags::None;
    if (InferredType->isConstantSize())
      Flags |= TypedMemoryCallsiteFlags::FixedSize;
    if (InferredType->isArray())
      Flags |= TypedMemoryCallsiteFlags::Array;
    if (InferredType->isPrefixedArray())
      Flags |= TypedMemoryCallsiteFlags::HeaderPrefixedArray;
    if (auto *PrefixedArrayType = InferredType->asPrefixedArray();
        PrefixedArrayType && !PrefixedArrayType->ElementType) {
      if (auto PrefixedArrayElement =
              Ctx.findFlexibleArrayElementType(PrefixedArrayType->HeaderType))
        InferredType = InferredAllocationType::createPrefixedArray(
            PrefixedArrayType->HeaderType, PrefixedArrayElement,
            PrefixedArrayType->IsConstantSize);
    }
    return RecordResult(SizeArg.IgnoreImplicit(), InferredType, Flags);
  }

  if (!ContainingCastExpr || !Call->getCallReturnType(Ctx)->isPointerType())
    return RecordResult(&SizeArg, std::nullopt, TypedMemoryCallsiteFlags::None);

  QualType CastDest = ContainingCastExpr->getType();
  if (!CastDest->isAnyPointerType())
    return RecordResult(&SizeArg, std::nullopt, TypedMemoryCallsiteFlags::None);

  QualType CastTarget = CastDest->getPointeeType();
  if (InferredType) {
    TypedMemoryCallsiteFlags Flags = TypedMemoryCallsiteFlags::None;
    if (InferredType->isConstantSize())
      Flags |= TypedMemoryCallsiteFlags::FixedSize;
    if (auto *InferredArray = InferredType->asArray()) {
      Flags |= TypedMemoryCallsiteFlags::Array;
      return RecordResult(ContainingCastExpr,
                          InferredAllocationType::createArray(
                              CastTarget, InferredArray->IsConstantSize),
                          Flags);
    }
    if (auto *PrefixedArray = InferredType->asPrefixedArray()) {
      Flags |= TypedMemoryCallsiteFlags::HeaderPrefixedArray;
      return RecordResult(ContainingCastExpr,
                          InferredAllocationType::createPrefixedArray(
                              CastDest->getPointeeType(),
                              PrefixedArray->ElementType,
                              PrefixedArray->IsConstantSize),
                          Flags);
    }
  }
  return RecordResult(
      ContainingCastExpr,
      InferredAllocationType::create(CastDest->getPointeeType()),
      TypedMemoryCallsiteFlags::FixedSize);
}

StringRef
ASTContext::getTypeSummaryDescription(TypedMemorySummary Summary) const {
  return getTMOInference().getTypeSummaryDescription(Summary);
}

ASTContext::TypedMemoryDescriptor
ASTContext::getTypedMemoryDescriptor(QualType QT, OverloadedOperatorKind Op,
                                     TypedMemoryCallsiteFlags Flags) const {
  return getTMOInference().getDescriptor(*this, QT, Op, Flags);
}

void ASTContext::setInferredInfoForCall(const CallExpr *Call,
                                        InferredTypeInfo Info) {
  getTMOInference().setInferredInfoForCall(Call, Info);
}

std::optional<InferredTypeInfo>
ASTContext::getInferredInfoForCall(const CallExpr *Call) const {
  return getTMOInference().getInferredInfoForCall(*this, Call);
}

InferredTypeInfo
ASTContext::inferTypedMemoryType(const CallExpr *Call, const Expr &SizeArg,
                                 const CastExpr *ContainingCastExpr) const {
  return getTMOInference().inferType(*this, Call, SizeArg, ContainingCastExpr);
}
