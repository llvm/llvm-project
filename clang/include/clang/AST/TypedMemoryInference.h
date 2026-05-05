#ifndef LLVM_CLANG_AST_TYPEDMEMORYINFERENCE_H
#define LLVM_CLANG_AST_TYPEDMEMORYINFERENCE_H

#include "clang/AST/Expr.h"
#include "clang/AST/TypedMemoryDescriptorBits.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <variant>

namespace clang {

class ASTContext;
class Type;

class InferredAllocationType {
public:
  enum class TypeKind {
    Object = 1,
    Array = 2,
    PrefixedArray = 3,
    Tuple = 4,
    UnknownLayout = 5,
  };

  // We recognize four patterns: a single object, arrays, header + array, and
  // tuples of multiple objects.
  struct Object {
    static const bool HasEntries = false;
    static constexpr TypeKind Kind = TypeKind::Object;
    bool isConstantSize() const { return true; }
    std::optional<QualType> primaryType() const { return Type; }
    QualType Type;

    void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
      RelatedTypes.push_back(Type);
    }
    void print(llvm::raw_ostream &OS,
               llvm::function_ref<void(QualType)> QuoteType) const;
  };

  struct Array {
    static constexpr TypeKind Kind = TypeKind::Array;
    static const bool HasEntries = false;
    bool isConstantSize() const { return IsConstantSize; }
    std::optional<QualType> primaryType() const { return ElementType; }
    bool IsConstantSize;
    QualType ElementType;
    void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
      RelatedTypes.push_back(ElementType);
    }
    void print(llvm::raw_ostream &OS,
               llvm::function_ref<void(QualType)> QuoteType) const;
  };

  struct PrefixedArray {
    static constexpr TypeKind Kind = TypeKind::PrefixedArray;
    static const bool HasEntries = false;
    bool isConstantSize() const { return IsConstantSize; }
    std::optional<QualType> primaryType() const { return HeaderType; }
    QualType HeaderType;
    bool IsConstantSize;
    // We don't necessarily know the actual element type, as we consider
    //   sizeof(T) + <some uninferrable expression>
    // to be a prefixed array
    std::optional<QualType> ElementType;
    void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
      RelatedTypes.push_back(HeaderType);
      if (ElementType)
        RelatedTypes.push_back(*ElementType);
    }
    void print(llvm::raw_ostream &OS,
               llvm::function_ref<void(QualType)> QuoteType) const;
  };

  struct Tuple {
    static constexpr TypeKind Kind = TypeKind::Tuple;
    static const bool HasEntries = true;
    bool isConstantSize() const { return true; }
    std::optional<QualType> primaryType() const { return Entries[0]; }
    SmallVector<QualType, 2> Entries;
    void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
      RelatedTypes.append(Entries);
    }
    void print(llvm::raw_ostream &OS,
               llvm::function_ref<void(QualType)> QuoteType) const;
  };

  // Pattern matching has failed so just accrue any types we've seen
  struct UnknownLayout {
    static constexpr TypeKind Kind = TypeKind::UnknownLayout;
    static const bool HasEntries = true;
    bool isConstantSize() const { return IsConstantSize; }
    std::optional<QualType> primaryType() const { return Entries[0]; }
    SmallVector<QualType, 2> Entries;
    bool IsConstantSize;
    void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
      RelatedTypes.append(Entries);
    }
    void print(llvm::raw_ostream &OS,
               llvm::function_ref<void(QualType)> QuoteType) const;
  };

private:
  using TypeUnionTy =
      std::variant<Object, Array, PrefixedArray, Tuple, UnknownLayout>;
  template <class FnType> auto visit(FnType &&F) const {
    return std::visit(F, TypeUnion);
  }
  TypeUnionTy TypeUnion;

public:
  TypeKind kind() const {
    return visit([](auto Type) { return decltype(Type)::Kind; });
  }

  bool isConstantSize() const {
    return visit([](auto &Type) { return Type.isConstantSize(); });
  }

  static InferredAllocationType create(QualType T) {
    InferredAllocationType IT;
    IT.TypeUnion = Object{T};
    return IT;
  }

  static InferredAllocationType createArray(QualType ElementType,
                                            bool IsConstantSize) {
    InferredAllocationType IT;
    IT.TypeUnion = Array{IsConstantSize, ElementType};
    return IT;
  }

  static InferredAllocationType
  createPrefixedArray(QualType HeaderType, std::optional<QualType> ElementType,
                      bool IsConstantSize) {
    InferredAllocationType IT;
    IT.TypeUnion = PrefixedArray{HeaderType, IsConstantSize, ElementType};
    return IT;
  }

  static InferredAllocationType
  createTuple(SmallVectorImpl<QualType> &&Entries) {
    InferredAllocationType IT;
    IT.TypeUnion = Tuple{std::move(Entries)};
    return IT;
  }

  static InferredAllocationType
  createUnknownLayout(SmallVectorImpl<QualType> &&Entries,
                      bool IsConstantSize) {
    InferredAllocationType Result;
    SmallVector<QualType, 2> Types;
    llvm::SmallDenseSet<QualType, 2> SeenTypes;
    for (QualType Type : Entries) {
      if (SeenTypes.insert(Type).second)
        Types.push_back(Type);
    }
    Result.TypeUnion = UnknownLayout{std::move(Types), IsConstantSize};
    return Result;
  }

  bool isPrefixedArray() const { return kind() == TypeKind::PrefixedArray; }
  bool isUnknownLayout() const { return kind() == TypeKind::UnknownLayout; }
  bool isArray() const { return kind() == TypeKind::Array; }

  bool hasPrefixedArrayElementType() const {
    auto *Prefixed = std::get_if<PrefixedArray>(&TypeUnion);
    return Prefixed && Prefixed->ElementType;
  }

  std::optional<QualType> prefixedArrayElementType() const {
    if (auto *Prefixed = std::get_if<PrefixedArray>(&TypeUnion))
      return Prefixed->ElementType;
    return std::nullopt;
  }

  const Object *asObject() const { return std::get_if<Object>(&TypeUnion); }
  const Array *asArray() const { return std::get_if<Array>(&TypeUnion); }
  const PrefixedArray *asPrefixedArray() const {
    return std::get_if<PrefixedArray>(&TypeUnion);
  }
  const UnknownLayout *asUnknownLayout() const {
    return std::get_if<UnknownLayout>(&TypeUnion);
  }
  const Tuple *asTuple() const { return std::get_if<Tuple>(&TypeUnion); }

  std::string describe(const ASTContext &Ctx) const;
  void print(llvm::raw_ostream &OS, const PrintingPolicy &Policy) const;

  void appendEntries(SmallVectorImpl<QualType> &Entries) const {
    if (hasEntries())
      Entries.append(entries().begin(), entries().end());
    else if (auto PT = primaryType())
      Entries.push_back(*PT);
    else
      llvm_unreachable("fixed-size type has no primaryType");
  }

  void extractRelatedTypes(SmallVectorImpl<QualType> &RelatedTypes) const {
    visit([&](auto &Type) { Type.extractRelatedTypes(RelatedTypes); });
  }

  bool hasEntries() const {
    return visit([](auto Type) { return decltype(Type)::HasEntries; });
  }
  ArrayRef<QualType> entries() const {
    return visit([](auto &Type) -> ArrayRef<QualType> {
      if constexpr (std::remove_reference_t<decltype(Type)>::HasEntries)
        return Type.Entries;
      else
        llvm_unreachable("Attempting to get entries from a non-entries type");
    });
  }
  std::optional<QualType> primaryType() const {
    return visit([](auto Type) { return Type.primaryType(); });
  }
};

struct InferredTypeInfo {
  std::optional<InferredAllocationType> Type;
  const Expr *InferenceSourceExpression;
  TypedMemoryCallsiteFlags InferredCallsiteFlags =
      TypedMemoryCallsiteFlags::None;
};

class TypedMemoryInference {
public:
  TypedMemoryInference() = default;

  struct TypedMemoryDescriptor {
    uint32_t IdentityHash;
    TypedMemorySummary Summary;
    // Only set if diagnostics are enabled.
    StringRef TypeDescription;

    TypedMemoryDescriptorBits asBits() const {
      TypedMemoryDescriptorBits Bits;
      Bits.Summary = Summary;
      Bits.Hash = IdentityHash;
      return Bits;
    }
  };

  TypedMemoryDescriptor getDescriptor(const ASTContext &Ctx, QualType QT,
                                      OverloadedOperatorKind,
                                      TypedMemoryCallsiteFlags);
  StringRef getTypeSummaryDescription(TypedMemorySummary Summary);
  void setInferredInfoForCall(const CallExpr *Call, InferredTypeInfo Info);
  std::optional<InferredTypeInfo>
  getInferredInfoForCall(const ASTContext &Ctx, const CallExpr *Call) const;
  InferredTypeInfo inferType(const ASTContext &Ctx, const CallExpr *Call,
                             const Expr &SizeArg,
                             const CastExpr *ContainingCastExpr);

private:
  TypedMemoryInference(const TypedMemoryInference &) = delete;
  TypedMemoryInference(TypedMemoryInference &&) = delete;

  using TypeDescriptorMapKeyTy =
      std::tuple<const Type *, TypedMemoryCallsiteFlags, TypedMemoryTypeKind>;
  struct TypedMemoryDescriptorMapValueTy {
    bool IsIncomplete;
    uint32_t IdentityHash;
    TypedMemorySummary Summary;
    std::unique_ptr<std::string> TypeDescription;
  };

  using TypeDescriptorMapTy =
      llvm::DenseMap<TypeDescriptorMapKeyTy, TypedMemoryDescriptorMapValueTy>;
  TypeDescriptorMapTy TypeDescriptorMap;
  using InferredTMOMapTy = llvm::DenseMap<const CallExpr *, InferredTypeInfo>;
  InferredTMOMapTy InferredTMOCallTypes;
  // Key is TypedMemorySummary::value()
  using TypeSummaryDescriptionCacheTy =
      llvm::DenseMap<uint32_t, std::unique_ptr<std::string>>;
  TypeSummaryDescriptionCacheTy TypeSummaryDescriptionCache;
};

} // namespace clang

#endif // LLVM_CLANG_AST_TYPEDMEMORYINFERENCE_H
