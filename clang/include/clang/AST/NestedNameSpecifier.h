//===- NestedNameSpecifier.h - C++ nested name specifiers -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the NestedNameSpecifier class, which represents
//  a C++ nested-name-specifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_NESTEDNAMESPECIFIER_H
#define LLVM_CLANG_AST_NESTEDNAMESPECIFIER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace clang {

class ASTContext;
class CXXRecordDecl;
class IdentifierInfo;
class LangOptions;
class NamespaceAliasDecl;
class NamespaceDecl;
struct PrintingPolicy;
class Type;
class TypeLoc;

/// \brief Represents a C++ nested name specifier, such as
/// "\::std::vector<int>::".
///
/// C++ nested name specifiers are the prefixes to qualified
/// names. For example, "foo::" in "foo::x" is a nested name
/// specifier. Nested name specifiers are made up of a sequence of
/// specifiers, each of which can be a namespace, type, identifier
/// (for dependent names), decltype specifier, or the global specifier ('::').
/// The last two specifiers can only appear at the start of a 
/// nested-namespace-specifier.
class NestedNameSpecifier : public llvm::FoldingSetNode {
  /// \brief Enumeration describing
  enum StoredSpecifierKind {
    StoredIdentifier = 0,
    StoredDecl = 1,
    StoredTypeSpec = 2,
    StoredTypeSpecWithTemplate = 3
  };

  /// \brief The nested name specifier that precedes this nested name
  /// specifier.
  ///
  /// The pointer is the nested-name-specifier that precedes this
  /// one. The integer stores one of the first four values of type
  /// SpecifierKind.
  llvm::PointerIntPair<NestedNameSpecifier *, 2, StoredSpecifierKind> Prefix;

  /// \brief The last component in the nested name specifier, which
  /// can be an identifier, a declaration, or a type.
  ///
  /// When the pointer is NULL, this specifier represents the global
  /// specifier '::'. Otherwise, the pointer is one of
  /// IdentifierInfo*, Namespace*, or Type*, depending on the kind of
  /// specifier as encoded within the prefix.
  void* Specifier = nullptr;

public:
  /// \brief The kind of specifier that completes this nested name
  /// specifier.
  enum SpecifierKind {
    /// \brief An identifier, stored as an IdentifierInfo*.
    Identifier,

    /// \brief A namespace, stored as a NamespaceDecl*.
    Namespace,

    /// \brief A namespace alias, stored as a NamespaceAliasDecl*.
    NamespaceAlias,

    /// \brief A type, stored as a Type*.
    TypeSpec,

    /// \brief A type that was preceded by the 'template' keyword,
    /// stored as a Type*.
    TypeSpecWithTemplate,

    /// \brief The global specifier '::'. There is no stored value.
    Global,

    /// \brief Microsoft's '__super' specifier, stored as a CXXRecordDecl* of
    /// the class it appeared in.
    Super
  };

private:
  /// \brief Builds the global specifier.
  NestedNameSpecifier() : Prefix(nullptr, StoredIdentifier) {}

  /// \brief Copy constructor used internally to clone nested name
  /// specifiers.
  NestedNameSpecifier(const NestedNameSpecifier &Other) = default;

  /// \brief Either find or insert the given nested name specifier
  /// mockup in the given context.
  static NestedNameSpecifier *FindOrInsert(const ASTContext &Context,
                                           const NestedNameSpecifier &Mockup);

public:
  NestedNameSpecifier &operator=(const NestedNameSpecifier &) = delete;

  /// \brief Builds a specifier combining a prefix and an identifier.
  ///
  /// The prefix must be dependent, since nested name specifiers
  /// referencing an identifier are only permitted when the identifier
  /// cannot be resolved.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     IdentifierInfo *II);

  /// \brief Builds a nested name specifier that names a namespace.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     const NamespaceDecl *NS);

  /// \brief Builds a nested name specifier that names a namespace alias.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     NamespaceAliasDecl *Alias);

  /// \brief Builds a nested name specifier that names a type.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     bool Template, const Type *T);

  /// \brief Builds a specifier that consists of just an identifier.
  ///
  /// The nested-name-specifier is assumed to be dependent, but has no
  /// prefix because the prefix is implied by something outside of the
  /// nested name specifier, e.g., in "x->Base::f", the "x" has a dependent
  /// type.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     IdentifierInfo *II);

  /// \brief Returns the nested name specifier representing the global
  /// scope.
  static NestedNameSpecifier *GlobalSpecifier(const ASTContext &Context);

  /// \brief Returns the nested name specifier representing the __super scope
  /// for the given CXXRecordDecl.
  static NestedNameSpecifier *SuperSpecifier(const ASTContext &Context,
                                             CXXRecordDecl *RD);

  /// \brief Return the prefix of this nested name specifier.
  ///
  /// The prefix contains all of the parts of the nested name
  /// specifier that preced this current specifier. For example, for a
  /// nested name specifier that represents "foo::bar::", the current
  /// specifier will contain "bar::" and the prefix will contain
  /// "foo::".
  NestedNameSpecifier *getPrefix() const { return Prefix.getPointer(); }

  /// \brief Determine what kind of nested name specifier is stored.
  SpecifierKind getKind() const;

  /// \brief Retrieve the identifier stored in this nested name
  /// specifier.
  IdentifierInfo *getAsIdentifier() const {
    if (Prefix.getInt() == StoredIdentifier)
      return (IdentifierInfo *)Specifier;

    return nullptr;
  }

  /// \brief Retrieve the namespace stored in this nested name
  /// specifier.
  NamespaceDecl *getAsNamespace() const;

  /// \brief Retrieve the namespace alias stored in this nested name
  /// specifier.
  NamespaceAliasDecl *getAsNamespaceAlias() const;

  /// \brief Retrieve the record declaration stored in this nested name
  /// specifier.
  CXXRecordDecl *getAsRecordDecl() const;

  /// \brief Retrieve the type stored in this nested name specifier.
  const Type *getAsType() const {
    if (Prefix.getInt() == StoredTypeSpec ||
        Prefix.getInt() == StoredTypeSpecWithTemplate)
      return (const Type *)Specifier;

    return nullptr;
  }

  /// \brief Whether this nested name specifier refers to a dependent
  /// type or not.
  bool isDependent() const;

  /// \brief Whether this nested name specifier involves a template
  /// parameter.
  bool isInstantiationDependent() const;

  /// \brief Whether this nested-name-specifier contains an unexpanded
  /// parameter pack (for C++11 variadic templates).
  bool containsUnexpandedParameterPack() const;

  /// \brief Print this nested name specifier to the given output
  /// stream.
  void print(raw_ostream &OS, const PrintingPolicy &Policy) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Prefix.getOpaqueValue());
    ID.AddPointer(Specifier);
  }

  /// \brief Dump the nested name specifier to standard output to aid
  /// in debugging.
  void dump(const LangOptions &LO) const;
  void dump() const;
};

/// \brief A C++ nested-name-specifier augmented with source location
/// information.
class NestedNameSpecifierLoc {
  NestedNameSpecifier *Qualifier = nullptr;
  void *Data = nullptr;

  /// \brief Determines the data length for the last component in the
  /// given nested-name-specifier.
  static unsigned getLocalDataLength(NestedNameSpecifier *Qualifier);

  /// \brief Determines the data length for the entire
  /// nested-name-specifier.
  static unsigned getDataLength(NestedNameSpecifier *Qualifier);

public:
  /// \brief Construct an empty nested-name-specifier.
  NestedNameSpecifierLoc() = default;

  /// \brief Construct a nested-name-specifier with source location information
  /// from
  NestedNameSpecifierLoc(NestedNameSpecifier *Qualifier, void *Data)
      : Qualifier(Qualifier), Data(Data) {}

  /// \brief Evaluates true when this nested-name-specifier location is
  /// non-empty.
  explicit operator bool() const { return Qualifier; }

  /// \brief Evaluates true when this nested-name-specifier location is
  /// empty.
  bool hasQualifier() const { return Qualifier; }

  /// \brief Retrieve the nested-name-specifier to which this instance
  /// refers.
  NestedNameSpecifier *getNestedNameSpecifier() const {
    return Qualifier;
  }

  /// \brief Retrieve the opaque pointer that refers to source-location data.
  void *getOpaqueData() const { return Data; }

  /// \brief Retrieve the source range covering the entirety of this
  /// nested-name-specifier.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::vector<int>::, the returned source range would cover
  /// from the initial '::' to the last '::'.
  SourceRange getSourceRange() const LLVM_READONLY;

  /// \brief Retrieve the source range covering just the last part of
  /// this nested-name-specifier, not including the prefix.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::vector<int>::, the returned source range would cover
  /// from "vector" to the last '::'.
  SourceRange getLocalSourceRange() const;

  /// \brief Retrieve the location of the beginning of this
  /// nested-name-specifier.
  SourceLocation getBeginLoc() const {
    return getSourceRange().getBegin();
  }

  /// \brief Retrieve the location of the end of this
  /// nested-name-specifier.
  SourceLocation getEndLoc() const {
    return getSourceRange().getEnd();
  }

  /// \brief Retrieve the location of the beginning of this
  /// component of the nested-name-specifier.
  SourceLocation getLocalBeginLoc() const {
    return getLocalSourceRange().getBegin();
  }

  /// \brief Retrieve the location of the end of this component of the
  /// nested-name-specifier.
  SourceLocation getLocalEndLoc() const {
    return getLocalSourceRange().getEnd();
  }

  /// \brief Return the prefix of this nested-name-specifier.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c \::std::vector<int>::, the prefix is \c \::std::. Note that the
  /// returned prefix may be empty, if this is the first component of
  /// the nested-name-specifier.
  NestedNameSpecifierLoc getPrefix() const {
    if (!Qualifier)
      return *this;

    return NestedNameSpecifierLoc(Qualifier->getPrefix(), Data);
  }

  /// \brief For a nested-name-specifier that refers to a type,
  /// retrieve the type with source-location information.
  TypeLoc getTypeLoc() const;

  /// \brief Determines the data length for the entire
  /// nested-name-specifier.
  unsigned getDataLength() const { return getDataLength(Qualifier); }

  friend bool operator==(NestedNameSpecifierLoc X,
                         NestedNameSpecifierLoc Y) {
    return X.Qualifier == Y.Qualifier && X.Data == Y.Data;
  }

  friend bool operator!=(NestedNameSpecifierLoc X,
                         NestedNameSpecifierLoc Y) {
    return !(X == Y);
  }
};

/// \brief Class that aids in the construction of nested-name-specifiers along
/// with source-location information for all of the components of the
/// nested-name-specifier.
class NestedNameSpecifierLocBuilder {
  /// \brief The current representation of the nested-name-specifier we're
  /// building.
  NestedNameSpecifier *Representation = nullptr;

  /// \brief Buffer used to store source-location information for the
  /// nested-name-specifier.
  ///
  /// Note that we explicitly manage the buffer (rather than using a
  /// SmallVector) because \c Declarator expects it to be possible to memcpy()
  /// a \c CXXScopeSpec, and CXXScopeSpec uses a NestedNameSpecifierLocBuilder.
  char *Buffer = nullptr;

  /// \brief The size of the buffer used to store source-location information
  /// for the nested-name-specifier.
  unsigned BufferSize = 0;

  /// \brief The capacity of the buffer used to store source-location
  /// information for the nested-name-specifier.
  unsigned BufferCapacity = 0;

public:
  NestedNameSpecifierLocBuilder() = default;
  NestedNameSpecifierLocBuilder(const NestedNameSpecifierLocBuilder &Other);

  NestedNameSpecifierLocBuilder &
  operator=(const NestedNameSpecifierLocBuilder &Other);

  ~NestedNameSpecifierLocBuilder() {
    if (BufferCapacity)
      free(Buffer);
  }

  /// \brief Retrieve the representation of the nested-name-specifier.
  NestedNameSpecifier *getRepresentation() const { return Representation; }

  /// \brief Extend the current nested-name-specifier by another
  /// nested-name-specifier component of the form 'type::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param TemplateKWLoc The location of the 'template' keyword, if present.
  ///
  /// \param TL The TypeLoc that describes the type preceding the '::'.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, SourceLocation TemplateKWLoc, TypeLoc TL,
              SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another
  /// nested-name-specifier component of the form 'identifier::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Identifier The identifier.
  ///
  /// \param IdentifierLoc The location of the identifier.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, IdentifierInfo *Identifier,
              SourceLocation IdentifierLoc, SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another
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
  void Extend(ASTContext &Context, NamespaceDecl *Namespace,
              SourceLocation NamespaceLoc, SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another
  /// nested-name-specifier component of the form 'namespace-alias::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Alias The namespace alias.
  ///
  /// \param AliasLoc The location of the namespace alias
  /// name.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, NamespaceAliasDecl *Alias,
              SourceLocation AliasLoc, SourceLocation ColonColonLoc);

  /// \brief Turn this (empty) nested-name-specifier into the global
  /// nested-name-specifier '::'.
  void MakeGlobal(ASTContext &Context, SourceLocation ColonColonLoc);
  
  /// \brief Turns this (empty) nested-name-specifier into '__super'
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
  void MakeSuper(ASTContext &Context, CXXRecordDecl *RD, 
                 SourceLocation SuperLoc, SourceLocation ColonColonLoc);

  /// \brief Make a new nested-name-specifier from incomplete source-location
  /// information.
  ///
  /// This routine should be used very, very rarely, in cases where we
  /// need to synthesize a nested-name-specifier. Most code should instead use
  /// \c Adopt() with a proper \c NestedNameSpecifierLoc.
  void MakeTrivial(ASTContext &Context, NestedNameSpecifier *Qualifier,
                   SourceRange R);

  /// \brief Adopt an existing nested-name-specifier (with source-range
  /// information).
  void Adopt(NestedNameSpecifierLoc Other);

  /// \brief Retrieve the source range covered by this nested-name-specifier.
  SourceRange getSourceRange() const LLVM_READONLY {
    return NestedNameSpecifierLoc(Representation, Buffer).getSourceRange();
  }

  /// \brief Retrieve a nested-name-specifier with location information,
  /// copied into the given AST context.
  ///
  /// \param Context The context into which this nested-name-specifier will be
  /// copied.
  NestedNameSpecifierLoc getWithLocInContext(ASTContext &Context) const;

  /// \brief Retrieve a nested-name-specifier with location
  /// information based on the information in this builder.
  ///
  /// This loc will contain references to the builder's internal data and may
  /// be invalidated by any change to the builder.
  NestedNameSpecifierLoc getTemporary() const {
    return NestedNameSpecifierLoc(Representation, Buffer);
  }

  /// \brief Clear out this builder, and prepare it to build another
  /// nested-name-specifier with source-location information.
  void Clear() {
    Representation = nullptr;
    BufferSize = 0;
  }

  /// \brief Retrieve the underlying buffer.
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
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           NestedNameSpecifier *NNS) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(NNS),
                  DiagnosticsEngine::ak_nestednamespec);
  return DB;
}

} // namespace clang

#endif // LLVM_CLANG_AST_NESTEDNAMESPECIFIER_H
