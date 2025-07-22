//===- NestedNameSpecifier.h - C++ nested name specifiers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the NestedNameSpecifier class, which represents
//  a C++ nested-name-specifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_NESTEDNAMESPECIFIERBASE_H
#define LLVM_CLANG_AST_NESTEDNAMESPECIFIERBASE_H

#include "clang/AST/DependenceFlags.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace clang {

class ASTContext;
class CXXRecordDecl;
class NamedDecl;
class IdentifierInfo;
class LangOptions;
class NamespaceBaseDecl;
struct PrintingPolicy;
class Type;
class TypeLoc;

struct NamespaceAndPrefix;
struct alignas(8) NamespaceAndPrefixStorage;

/// Represents a C++ nested name specifier, such as
/// "\::std::vector<int>::".
///
/// C++ nested name specifiers are the prefixes to qualified
/// names. For example, "foo::" in "foo::x" is a nested name
/// specifier. Nested name specifiers are made up of a sequence of
/// specifiers, each of which can be a namespace, type, decltype specifier, or
/// the global specifier ('::'). The last two specifiers can only appear at the
/// start of a nested-namespace-specifier.
class NestedNameSpecifier {
  enum class FlagKind { Null, Global, Invalid };
  enum class StoredKind {
    Type,
    NamespaceOrSuper,
    NamespaceWithGlobal,
    NamespaceWithNamespace
  };
  static constexpr uintptr_t FlagBits = 2, FlagMask = (1u << FlagBits) - 1u,
                             FlagOffset = 1, PtrOffset = FlagBits + FlagOffset,
                             PtrMask = (1u << PtrOffset) - 1u;

  uintptr_t StoredOrFlag;

  explicit NestedNameSpecifier(uintptr_t StoredOrFlag)
      : StoredOrFlag(StoredOrFlag) {}
  struct PtrKind {
    StoredKind SK;
    const void *Ptr;
  };
  explicit NestedNameSpecifier(PtrKind PK)
      : StoredOrFlag(uintptr_t(PK.Ptr) | (uintptr_t(PK.SK) << FlagOffset)) {
    assert(PK.Ptr != nullptr);
    assert((uintptr_t(PK.Ptr) & ((1u << PtrOffset) - 1u)) == 0);
    assert((uintptr_t(PK.Ptr) >> PtrOffset) != 0);
  }

  explicit constexpr NestedNameSpecifier(FlagKind K)
      : StoredOrFlag(uintptr_t(K) << FlagOffset) {}

  bool isStoredKind() const { return (StoredOrFlag >> PtrOffset) != 0; }

  std::pair<StoredKind, const void *> getStored() const {
    assert(isStoredKind());
    return {StoredKind(StoredOrFlag >> FlagOffset & FlagMask),
            reinterpret_cast<const void *>(StoredOrFlag & ~PtrMask)};
  }

  FlagKind getFlagKind() const {
    assert(!isStoredKind());
    return FlagKind(StoredOrFlag >> FlagOffset);
  }

  static const NamespaceAndPrefixStorage *
  MakeNamespaceAndPrefixStorage(const ASTContext &Ctx,
                                const NamespaceBaseDecl *Namespace,
                                NestedNameSpecifier Prefix);
  static inline PtrKind MakeNamespacePtrKind(const ASTContext &Ctx,
                                             const NamespaceBaseDecl *Namespace,
                                             NestedNameSpecifier Prefix);

public:
  static constexpr NestedNameSpecifier getInvalid() {
    return NestedNameSpecifier(FlagKind::Invalid);
  }

  static constexpr NestedNameSpecifier getGlobal() {
    return NestedNameSpecifier(FlagKind::Global);
  }

  NestedNameSpecifier() : NestedNameSpecifier(FlagKind::Invalid) {}

  /// The kind of specifier that completes this nested name
  /// specifier.
  enum class Kind {
    /// Empty.
    Null,

    /// The global specifier '::'. There is no stored value.
    Global,

    /// A type, stored as a Type*.
    Type,

    /// A namespace-like entity, stored as a NamespaceBaseDecl*.
    Namespace,

    /// Microsoft's '__super' specifier, stored as a CXXRecordDecl* of
    /// the class it appeared in.
    MicrosoftSuper,
  };

  inline Kind getKind() const;

  NestedNameSpecifier(std::nullopt_t) : StoredOrFlag(0) {}

  explicit inline NestedNameSpecifier(const Type *T);

  /// Builds a nested name specifier that names a namespace.
  inline NestedNameSpecifier(const ASTContext &Ctx,
                             const NamespaceBaseDecl *Namespace,
                             NestedNameSpecifier Prefix);

  /// Builds a nested name specifier that names a class through microsoft's
  /// __super specifier.
  explicit inline NestedNameSpecifier(CXXRecordDecl *RD);

  explicit operator bool() const { return StoredOrFlag != 0; }

  void *getAsVoidPointer() const {
    return reinterpret_cast<void *>(StoredOrFlag);
  }
  static NestedNameSpecifier getFromVoidPointer(const void *Ptr) {
    return NestedNameSpecifier(reinterpret_cast<uintptr_t>(Ptr));
  }

  const Type *getAsType() const {
    auto [Kind, Ptr] = getStored();
    assert(Kind == StoredKind::Type);
    assert(Ptr != nullptr);
    return static_cast<const Type *>(Ptr);
  }

  inline NamespaceAndPrefix getAsNamespaceAndPrefix() const;

  CXXRecordDecl *getAsMicrosoftSuper() const {
    auto [Kind, Ptr] = getStored();
    assert(Kind == StoredKind::NamespaceOrSuper);
    assert(Ptr != nullptr);
    return static_cast<CXXRecordDecl *>(const_cast<void *>(Ptr));
  }

  /// Retrieve the record declaration stored in this nested name
  /// specifier, or null.
  inline CXXRecordDecl *getAsRecordDecl() const;

  friend bool operator==(NestedNameSpecifier LHS, NestedNameSpecifier RHS) {
    return LHS.StoredOrFlag == RHS.StoredOrFlag;
  }
  friend bool operator!=(NestedNameSpecifier LHS, NestedNameSpecifier RHS) {
    return LHS.StoredOrFlag != RHS.StoredOrFlag;
  }

  /// Retrieves the "canonical" nested name specifier for a
  /// given nested name specifier.
  ///
  /// The canonical nested name specifier is a nested name specifier
  /// that uniquely identifies a type or namespace within the type
  /// system. For example, given:
  ///
  /// \code
  /// namespace N {
  ///   struct S {
  ///     template<typename T> struct X { typename T* type; };
  ///   };
  /// }
  ///
  /// template<typename T> struct Y {
  ///   typename N::S::X<T>::type member;
  /// };
  /// \endcode
  ///
  /// Here, the nested-name-specifier for N::S::X<T>:: will be
  /// S::X<template-param-0-0>, since 'S' and 'X' are uniquely defined
  /// by declarations in the type system and the canonical type for
  /// the template type parameter 'T' is template-param-0-0.
  inline NestedNameSpecifier getCanonical() const;

  /// Whether this nested name specifier is canonical.
  inline bool isCanonical() const;

  /// Whether this nested name specifier starts with a '::'.
  bool isFullyQualified() const;

  NestedNameSpecifierDependence getDependence() const;

  /// Whether this nested name specifier refers to a dependent
  /// type or not.
  bool isDependent() const {
    return getDependence() & NestedNameSpecifierDependence::Dependent;
  }

  /// Whether this nested name specifier involves a template
  /// parameter.
  bool isInstantiationDependent() const {
    return getDependence() & NestedNameSpecifierDependence::Instantiation;
  }

  /// Whether this nested-name-specifier contains an unexpanded
  /// parameter pack (for C++11 variadic templates).
  bool containsUnexpandedParameterPack() const {
    return getDependence() & NestedNameSpecifierDependence::UnexpandedPack;
  }

  /// Whether this nested name specifier contains an error.
  bool containsErrors() const {
    return getDependence() & NestedNameSpecifierDependence::Error;
  }

  /// Print this nested name specifier to the given output stream. If
  /// `ResolveTemplateArguments` is true, we'll print actual types, e.g.
  /// `ns::SomeTemplate<int, MyClass>` instead of
  /// `ns::SomeTemplate<Container::value_type, T>`.
  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             bool ResolveTemplateArguments = false,
             bool PrintFinalScopeResOp = true) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(StoredOrFlag);
  }

  /// Dump the nested name specifier to aid in debugging.
  void dump(llvm::raw_ostream *OS = nullptr,
            const LangOptions *LO = nullptr) const;
  void dump(const LangOptions &LO) const;
  void dump(llvm::raw_ostream &OS) const;
  void dump(llvm::raw_ostream &OS, const LangOptions &LO) const;

  static constexpr auto NumLowBitsAvailable = FlagOffset;
};

struct NamespaceAndPrefix {
  const NamespaceBaseDecl *Namespace;
  NestedNameSpecifier Prefix;
};

struct alignas(8) NamespaceAndPrefixStorage : NamespaceAndPrefix,
                                              llvm::FoldingSetNode {
  NamespaceAndPrefixStorage(const NamespaceBaseDecl *Namespace,
                            NestedNameSpecifier Prefix)
      : NamespaceAndPrefix{Namespace, Prefix} {}
  void Profile(llvm::FoldingSetNodeID &ID) { Profile(ID, Namespace, Prefix); }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const NamespaceBaseDecl *Namespace,
                      NestedNameSpecifier Prefix) {
    ID.AddPointer(Namespace);
    Prefix.Profile(ID);
  }
};

NamespaceAndPrefix NestedNameSpecifier::getAsNamespaceAndPrefix() const {
  auto [Kind, Ptr] = getStored();
  switch (Kind) {
  case StoredKind::NamespaceOrSuper:
  case StoredKind::NamespaceWithGlobal:
    return {static_cast<const NamespaceBaseDecl *>(Ptr),
            Kind == StoredKind::NamespaceWithGlobal
                ? NestedNameSpecifier::getGlobal()
                : std::nullopt};
  case StoredKind::NamespaceWithNamespace:
    return *static_cast<const NamespaceAndPrefixStorage *>(Ptr);
  case StoredKind::Type:;
  }
  llvm_unreachable("unexpected stored kind");
}

struct NamespaceAndPrefixLoc;

/// A C++ nested-name-specifier augmented with source location
/// information.
class NestedNameSpecifierLoc {
  NestedNameSpecifier Qualifier = std::nullopt;
  void *Data = nullptr;

  /// Load a (possibly unaligned) source location from a given address
  /// and offset.
  SourceLocation LoadSourceLocation(unsigned Offset) const {
    SourceLocation::UIntTy Raw;
    memcpy(&Raw, static_cast<char *>(Data) + Offset, sizeof(Raw));
    return SourceLocation::getFromRawEncoding(Raw);
  }

  /// Load a (possibly unaligned) pointer from a given address and
  /// offset.
  void *LoadPointer(unsigned Offset) const {
    void *Result;
    memcpy(&Result, static_cast<char *>(Data) + Offset, sizeof(void *));
    return Result;
  }

  /// Determines the data length for the last component in the
  /// given nested-name-specifier.
  static inline unsigned getLocalDataLength(NestedNameSpecifier Qualifier);

  /// Determines the data length for the entire
  /// nested-name-specifier.
  static inline unsigned getDataLength(NestedNameSpecifier Qualifier);

public:
  /// Construct an empty nested-name-specifier.
  NestedNameSpecifierLoc() = default;

  /// Construct a nested-name-specifier with source location information
  /// from
  NestedNameSpecifierLoc(NestedNameSpecifier Qualifier, void *Data)
      : Qualifier(Qualifier), Data(Data) {}

  /// Evaluates true when this nested-name-specifier location is
  /// non-empty.
  explicit operator bool() const { return bool(Qualifier); }

  /// Evaluates true when this nested-name-specifier location is
  /// non-empty.
  bool hasQualifier() const { return bool(Qualifier); }

  /// Retrieve the nested-name-specifier to which this instance
  /// refers.
  NestedNameSpecifier getNestedNameSpecifier() const { return Qualifier; }

  /// Retrieve the opaque pointer that refers to source-location data.
  void *getOpaqueData() const { return Data; }

  /// Retrieve the source range covering the entirety of this
  /// nested-name-specifier.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::vector<int>::, the returned source range would cover
  /// from the initial '::' to the last '::'.
  inline SourceRange getSourceRange() const LLVM_READONLY;

  /// Retrieve the source range covering just the last part of
  /// this nested-name-specifier, not including the prefix.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::vector<int>::, the returned source range would cover
  /// from "vector" to the last '::'.
  inline SourceRange getLocalSourceRange() const;

  /// Retrieve the location of the beginning of this
  /// nested-name-specifier.
  SourceLocation getBeginLoc() const;

  /// Retrieve the location of the end of this
  /// nested-name-specifier.
  inline SourceLocation getEndLoc() const;

  /// Retrieve the location of the beginning of this
  /// component of the nested-name-specifier.
  inline SourceLocation getLocalBeginLoc() const;

  /// Retrieve the location of the end of this component of the
  /// nested-name-specifier.
  inline SourceLocation getLocalEndLoc() const;

  /// For a nested-name-specifier that refers to a namespace,
  /// retrieve the namespace and its prefix.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::chrono::, the prefix is \c \::std::. Note that the
  /// returned prefix may be empty, if this is the first component of
  /// the nested-name-specifier.
  inline NamespaceAndPrefixLoc castAsNamespaceAndPrefix() const;
  inline NamespaceAndPrefixLoc getAsNamespaceAndPrefix() const;

  /// For a nested-name-specifier that refers to a type,
  /// retrieve the type with source-location information.
  inline TypeLoc castAsTypeLoc() const;
  inline TypeLoc getAsTypeLoc() const;

  /// Determines the data length for the entire
  /// nested-name-specifier.
  inline unsigned getDataLength() const;

  friend bool operator==(NestedNameSpecifierLoc X, NestedNameSpecifierLoc Y) {
    return X.Qualifier == Y.Qualifier && X.Data == Y.Data;
  }

  friend bool operator!=(NestedNameSpecifierLoc X, NestedNameSpecifierLoc Y) {
    return !(X == Y);
  }
};

struct NamespaceAndPrefixLoc {
  const NamespaceBaseDecl *Namespace = nullptr;
  NestedNameSpecifierLoc Prefix;

  explicit operator bool() const { return Namespace != nullptr; }
};

/// Class that aids in the construction of nested-name-specifiers along
/// with source-location information for all of the components of the
/// nested-name-specifier.
class NestedNameSpecifierLocBuilder {
  /// The current representation of the nested-name-specifier we're
  /// building.
  NestedNameSpecifier Representation = std::nullopt;

  /// Buffer used to store source-location information for the
  /// nested-name-specifier.
  ///
  /// Note that we explicitly manage the buffer (rather than using a
  /// SmallVector) because \c Declarator expects it to be possible to memcpy()
  /// a \c CXXScopeSpec, and CXXScopeSpec uses a NestedNameSpecifierLocBuilder.
  char *Buffer = nullptr;

  /// The size of the buffer used to store source-location information
  /// for the nested-name-specifier.
  unsigned BufferSize = 0;

  /// The capacity of the buffer used to store source-location
  /// information for the nested-name-specifier.
  unsigned BufferCapacity = 0;

  void PushTrivial(ASTContext &Context, NestedNameSpecifier Qualifier,
                   SourceRange R);

public:
  NestedNameSpecifierLocBuilder() = default;
  NestedNameSpecifierLocBuilder(const NestedNameSpecifierLocBuilder &Other);

  NestedNameSpecifierLocBuilder &
  operator=(const NestedNameSpecifierLocBuilder &Other);

  ~NestedNameSpecifierLocBuilder() {
    if (BufferCapacity)
      free(Buffer);
  }

  /// Retrieve the representation of the nested-name-specifier.
  NestedNameSpecifier getRepresentation() const { return Representation; }

  /// Make a nested-name-specifier of the form 'type::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param TL The TypeLoc that describes the type preceding the '::'.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Make(ASTContext &Context, TypeLoc TL, SourceLocation ColonColonLoc);

  /// Extend the current nested-name-specifier by another
  /// nested-name-specifier component of the form 'namespace::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Namespace The namespace.
  ///
  /// \param NamespaceLoc The location of the namespace name.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, const NamespaceBaseDecl *Namespace,
              SourceLocation NamespaceLoc, SourceLocation ColonColonLoc);

  /// Turn this (empty) nested-name-specifier into the global
  /// nested-name-specifier '::'.
  void MakeGlobal(ASTContext &Context, SourceLocation ColonColonLoc);

  /// Turns this (empty) nested-name-specifier into '__super'
  /// nested-name-specifier.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param RD The declaration of the class in which nested-name-specifier
  /// appeared.
  ///
  /// \param SuperLoc The location of the '__super' keyword.
  /// name.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void MakeMicrosoftSuper(ASTContext &Context, CXXRecordDecl *RD,
                          SourceLocation SuperLoc,
                          SourceLocation ColonColonLoc);

  /// Make a new nested-name-specifier from incomplete source-location
  /// information.
  ///
  /// This routine should be used very, very rarely, in cases where we
  /// need to synthesize a nested-name-specifier. Most code should instead use
  /// \c Adopt() with a proper \c NestedNameSpecifierLoc.
  void MakeTrivial(ASTContext &Context, NestedNameSpecifier Qualifier,
                   SourceRange R) {
    Representation = Qualifier;
    BufferSize = 0;
    PushTrivial(Context, Qualifier, R);
  }

  /// Adopt an existing nested-name-specifier (with source-range
  /// information).
  void Adopt(NestedNameSpecifierLoc Other);

  /// Retrieve the source range covered by this nested-name-specifier.
  inline SourceRange getSourceRange() const LLVM_READONLY;

  /// Retrieve a nested-name-specifier with location information,
  /// copied into the given AST context.
  ///
  /// \param Context The context into which this nested-name-specifier will be
  /// copied.
  NestedNameSpecifierLoc getWithLocInContext(ASTContext &Context) const;

  /// Retrieve a nested-name-specifier with location
  /// information based on the information in this builder.
  ///
  /// This loc will contain references to the builder's internal data and may
  /// be invalidated by any change to the builder.
  NestedNameSpecifierLoc getTemporary() const {
    return NestedNameSpecifierLoc(Representation, Buffer);
  }

  /// Clear out this builder, and prepare it to build another
  /// nested-name-specifier with source-location information.
  void Clear() {
    Representation = std::nullopt;
    BufferSize = 0;
  }

  /// Retrieve the underlying buffer.
  ///
  /// \returns A pair containing a pointer to the buffer of source-location
  /// data and the size of the source-location data that resides in that
  /// buffer.
  std::pair<char *, unsigned> getBuffer() const {
    return std::make_pair(Buffer, BufferSize);
  }
};

/// Insertion operator for diagnostics.  This allows sending
/// NestedNameSpecifiers into a diagnostic with <<.
inline const StreamingDiagnostic &operator<<(const StreamingDiagnostic &DB,
                                             NestedNameSpecifier NNS) {
  DB.AddTaggedVal(reinterpret_cast<uintptr_t>(NNS.getAsVoidPointer()),
                  DiagnosticsEngine::ak_nestednamespec);
  return DB;
}

} // namespace clang

namespace llvm {

template <> struct PointerLikeTypeTraits<clang::NestedNameSpecifier> {
  static void *getAsVoidPointer(clang::NestedNameSpecifier P) {
    return P.getAsVoidPointer();
  }
  static clang::NestedNameSpecifier getFromVoidPointer(const void *P) {
    return clang::NestedNameSpecifier::getFromVoidPointer(P);
  }
  static constexpr int NumLowBitsAvailable =
      clang::NestedNameSpecifier::NumLowBitsAvailable;
};

} // namespace llvm

#endif // LLVM_CLANG_AST_NESTEDNAMESPECIFIERBASE_H
