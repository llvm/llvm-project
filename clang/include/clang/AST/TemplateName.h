//===- TemplateName.h - C++ Template Name Representation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TemplateName interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEMPLATENAME_H
#define LLVM_CLANG_AST_TEMPLATENAME_H

#include "clang/AST/DependenceFlags.h"
#include "clang/AST/NestedNameSpecifierBase.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/UnsignedOrNone.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <cassert>
#include <optional>

namespace clang {

class ASTContext;
class Decl;
class DependentTemplateName;
class IdentifierInfo;
class NamedDecl;
class NestedNameSpecifier;
enum OverloadedOperatorKind : int;
class OverloadedTemplateStorage;
class AssumedTemplateStorage;
class DeducedTemplateStorage;
struct PrintingPolicy;
class QualifiedTemplateName;
class SubstTemplateTemplateParmPackStorage;
class SubstTemplateTemplateParmStorage;
class TemplateArgument;
class TemplateDecl;
class TemplateTemplateParmDecl;
class UsingShadowDecl;

/// Implementation class used to describe either a set of overloaded
/// template names or an already-substituted template template parameter pack.
class UncommonTemplateNameStorage {
protected:
  enum Kind {
    Overloaded,
    Assumed, // defined in DeclarationName.h
    Deduced,
    SubstTemplateTemplateParm,
    SubstTemplateTemplateParmPack
  };

  struct BitsTag {
    LLVM_PREFERRED_TYPE(Kind)
    unsigned Kind : 3;

    // The template parameter index.
    unsigned Index : 14;

    /// The pack index, or the number of stored templates
    /// or template arguments, depending on which subclass we have.
    unsigned Data : 15;
  };

  union {
    struct BitsTag Bits;
    void *PointerAlignment;
  };

  UncommonTemplateNameStorage(Kind Kind, unsigned Index, unsigned Data) {
    Bits.Kind = Kind;
    Bits.Index = Index;
    Bits.Data = Data;
  }

public:
  OverloadedTemplateStorage *getAsOverloadedStorage()  {
    return Bits.Kind == Overloaded
             ? reinterpret_cast<OverloadedTemplateStorage *>(this)
             : nullptr;
  }

  AssumedTemplateStorage *getAsAssumedTemplateName()  {
    return Bits.Kind == Assumed
             ? reinterpret_cast<AssumedTemplateStorage *>(this)
             : nullptr;
  }

  DeducedTemplateStorage *getAsDeducedTemplateName() {
    return Bits.Kind == Deduced
               ? reinterpret_cast<DeducedTemplateStorage *>(this)
               : nullptr;
  }

  SubstTemplateTemplateParmStorage *getAsSubstTemplateTemplateParm() {
    return Bits.Kind == SubstTemplateTemplateParm
             ? reinterpret_cast<SubstTemplateTemplateParmStorage *>(this)
             : nullptr;
  }

  SubstTemplateTemplateParmPackStorage *getAsSubstTemplateTemplateParmPack() {
    return Bits.Kind == SubstTemplateTemplateParmPack
             ? reinterpret_cast<SubstTemplateTemplateParmPackStorage *>(this)
             : nullptr;
  }
};

/// A structure for storing the information associated with an
/// overloaded template name.
class OverloadedTemplateStorage : public UncommonTemplateNameStorage {
  friend class ASTContext;

  OverloadedTemplateStorage(unsigned size)
      : UncommonTemplateNameStorage(Overloaded, 0, size) {}

  NamedDecl **getStorage() {
    return reinterpret_cast<NamedDecl **>(this + 1);
  }
  NamedDecl * const *getStorage() const {
    return reinterpret_cast<NamedDecl *const *>(this + 1);
  }

public:
  unsigned size() const { return Bits.Data; }

  using iterator = NamedDecl *const *;

  iterator begin() const { return getStorage(); }
  iterator end() const { return getStorage() + Bits.Data; }

  llvm::ArrayRef<NamedDecl*> decls() const {
    return llvm::ArrayRef(begin(), end());
  }
};

/// A structure for storing an already-substituted template template
/// parameter pack.
///
/// This kind of template names occurs when the parameter pack has been
/// provided with a template template argument pack in a context where its
/// enclosing pack expansion could not be fully expanded.
class SubstTemplateTemplateParmPackStorage : public UncommonTemplateNameStorage,
                                             public llvm::FoldingSetNode {
  const TemplateArgument *Arguments;
  llvm::PointerIntPair<Decl *, 1, bool> AssociatedDeclAndFinal;

public:
  SubstTemplateTemplateParmPackStorage(ArrayRef<TemplateArgument> ArgPack,
                                       Decl *AssociatedDecl, unsigned Index,
                                       bool Final);

  /// A template-like entity which owns the whole pattern being substituted.
  /// This will own a set of template parameters.
  Decl *getAssociatedDecl() const;

  /// Returns the index of the replaced parameter in the associated declaration.
  /// This should match the result of `getParameterPack()->getIndex()`.
  unsigned getIndex() const { return Bits.Index; }

  // When true the substitution will be 'Final' (subst node won't be placed).
  bool getFinal() const;

  /// Retrieve the template template parameter pack being substituted.
  TemplateTemplateParmDecl *getParameterPack() const;

  /// Retrieve the template template argument pack with which this
  /// parameter was substituted.
  TemplateArgument getArgumentPack() const;

  void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context);

  static void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                      const TemplateArgument &ArgPack, Decl *AssociatedDecl,
                      unsigned Index, bool Final);
};

struct DefaultArguments {
  // The position in the template parameter list
  // the first argument corresponds to.
  unsigned StartPos;
  ArrayRef<TemplateArgument> Args;

  operator bool() const { return !Args.empty(); }
};

/// Represents a C++ template name within the type system.
///
/// A C++ template name refers to a template within the C++ type
/// system. In most cases, a template name is simply a reference to a
/// class template, e.g.
///
/// \code
/// template<typename T> class X { };
///
/// X<int> xi;
/// \endcode
///
/// Here, the 'X' in \c X<int> is a template name that refers to the
/// declaration of the class template X, above. Template names can
/// also refer to function templates, C++0x template aliases, etc.
///
/// Some template names are dependent. For example, consider:
///
/// \code
/// template<typename MetaFun, typename T1, typename T2> struct apply2 {
///   typedef typename MetaFun::template apply<T1, T2>::type type;
/// };
/// \endcode
///
/// Here, "apply" is treated as a template name within the typename
/// specifier in the typedef. "apply" is a nested template, and can
/// only be understood in the context of a template instantiation,
/// hence is represented as a dependent template name.
class TemplateName {
  // NameDecl is either a TemplateDecl or a UsingShadowDecl depending on the
  // NameKind.
  // !! There is no free low bits in 32-bit builds to discriminate more than 4
  // pointer types in PointerUnion.
  using StorageType =
      llvm::PointerUnion<Decl *, UncommonTemplateNameStorage *,
                         QualifiedTemplateName *, DependentTemplateName *>;

  StorageType Storage;

  explicit TemplateName(void *Ptr);

public:
  // Kind of name that is actually stored.
  enum NameKind {
    /// A single template declaration.
    Template,

    /// A set of overloaded template declarations.
    OverloadedTemplate,

    /// An unqualified-id that has been assumed to name a function template
    /// that will be found by ADL.
    AssumedTemplate,

    /// A qualified template name, where the qualification is kept
    /// to describe the source code as written.
    QualifiedTemplate,

    /// A dependent template name that has not been resolved to a
    /// template (or set of templates).
    DependentTemplate,

    /// A template template parameter that has been substituted
    /// for some other template name.
    SubstTemplateTemplateParm,

    /// A template template parameter pack that has been substituted for
    /// a template template argument pack, but has not yet been expanded into
    /// individual arguments.
    SubstTemplateTemplateParmPack,

    /// A template name that refers to a template declaration found through a
    /// specific using shadow declaration.
    UsingTemplate,

    /// A template name that refers to another TemplateName with deduced default
    /// arguments.
    DeducedTemplate,
  };

  TemplateName() = default;
  explicit TemplateName(TemplateDecl *Template);
  explicit TemplateName(OverloadedTemplateStorage *Storage);
  explicit TemplateName(AssumedTemplateStorage *Storage);
  explicit TemplateName(SubstTemplateTemplateParmStorage *Storage);
  explicit TemplateName(SubstTemplateTemplateParmPackStorage *Storage);
  explicit TemplateName(QualifiedTemplateName *Qual);
  explicit TemplateName(DependentTemplateName *Dep);
  explicit TemplateName(UsingShadowDecl *Using);
  explicit TemplateName(DeducedTemplateStorage *Deduced);

  /// Determine whether this template name is NULL.
  bool isNull() const;

  // Get the kind of name that is actually stored.
  NameKind getKind() const;

  /// Retrieve the underlying template declaration that
  /// this template name refers to, if known.
  ///
  /// \returns The template declaration that this template name refers
  /// to, if any. If the template name does not refer to a specific
  /// declaration because it is a dependent name, or if it refers to a
  /// set of function templates, returns NULL.
  TemplateDecl *getAsTemplateDecl(bool IgnoreDeduced = false) const;

  /// Retrieves the underlying template name that
  /// this template name refers to, along with the
  /// deduced default arguments, if any.
  std::pair<TemplateName, DefaultArguments>
  getTemplateDeclAndDefaultArgs() const;

  /// Retrieve the underlying, overloaded function template
  /// declarations that this template name refers to, if known.
  ///
  /// \returns The set of overloaded function templates that this template
  /// name refers to, if known. If the template name does not refer to a
  /// specific set of function templates because it is a dependent name or
  /// refers to a single template, returns NULL.
  OverloadedTemplateStorage *getAsOverloadedTemplate() const;

  /// Retrieve information on a name that has been assumed to be a
  /// template-name in order to permit a call via ADL.
  AssumedTemplateStorage *getAsAssumedTemplateName() const;

  /// Retrieve the substituted template template parameter, if
  /// known.
  ///
  /// \returns The storage for the substituted template template parameter,
  /// if known. Otherwise, returns NULL.
  SubstTemplateTemplateParmStorage *getAsSubstTemplateTemplateParm() const;

  /// Retrieve the substituted template template parameter pack, if
  /// known.
  ///
  /// \returns The storage for the substituted template template parameter pack,
  /// if known. Otherwise, returns NULL.
  SubstTemplateTemplateParmPackStorage *
  getAsSubstTemplateTemplateParmPack() const;

  /// Retrieve the underlying qualified template name
  /// structure, if any.
  QualifiedTemplateName *getAsQualifiedTemplateName() const;

  /// Retrieve the underlying dependent template name
  /// structure, if any.
  DependentTemplateName *getAsDependentTemplateName() const;

  // Retrieve the qualifier and template keyword stored in either a underlying
  // DependentTemplateName or QualifiedTemplateName.
  std::tuple<NestedNameSpecifier, bool> getQualifierAndTemplateKeyword() const;

  NestedNameSpecifier getQualifier() const {
    return std::get<0>(getQualifierAndTemplateKeyword());
  }

  /// Retrieve the using shadow declaration through which the underlying
  /// template declaration is introduced, if any.
  UsingShadowDecl *getAsUsingShadowDecl() const;

  /// Retrieve the deduced template info, if any.
  DeducedTemplateStorage *getAsDeducedTemplateName() const;

  std::optional<TemplateName> desugar(bool IgnoreDeduced) const;

  TemplateName getUnderlying() const;

  TemplateNameDependence getDependence() const;

  /// Determines whether this is a dependent template name.
  bool isDependent() const;

  /// Determines whether this is a template name that somehow
  /// depends on a template parameter.
  bool isInstantiationDependent() const;

  /// Determines whether this template name contains an
  /// unexpanded parameter pack (for C++0x variadic templates).
  bool containsUnexpandedParameterPack() const;

  enum class Qualified { None, AsWritten };
  /// Print the template name.
  ///
  /// \param OS the output stream to which the template name will be
  /// printed.
  ///
  /// \param Qual print the (Qualified::None) simple name,
  /// (Qualified::AsWritten) any written (possibly partial) qualifier, or
  /// (Qualified::Fully) the fully qualified name.
  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             Qualified Qual = Qualified::AsWritten) const;

  /// Debugging aid that dumps the template name.
  void dump(raw_ostream &OS, const ASTContext &Context) const;

  /// Debugging aid that dumps the template name to standard
  /// error.
  void dump() const;

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddPointer(Storage.getOpaqueValue());
  }

  /// Retrieve the template name as a void pointer.
  void *getAsVoidPointer() const { return Storage.getOpaqueValue(); }

  /// Build a template name from a void pointer.
  static TemplateName getFromVoidPointer(void *Ptr) {
    return TemplateName(Ptr);
  }

  /// Structural equality.
  bool operator==(TemplateName Other) const { return Storage == Other.Storage; }
  bool operator!=(TemplateName Other) const { return !operator==(Other); }
};

/// Insertion operator for diagnostics.  This allows sending TemplateName's
/// into a diagnostic with <<.
const StreamingDiagnostic &operator<<(const StreamingDiagnostic &DB,
                                      TemplateName N);

/// A structure for storing the information associated with a
/// substituted template template parameter.
class SubstTemplateTemplateParmStorage
  : public UncommonTemplateNameStorage, public llvm::FoldingSetNode {
  friend class ASTContext;

  TemplateName Replacement;
  Decl *AssociatedDecl;

  SubstTemplateTemplateParmStorage(TemplateName Replacement,
                                   Decl *AssociatedDecl, unsigned Index,
                                   UnsignedOrNone PackIndex, bool Final)
      : UncommonTemplateNameStorage(
            SubstTemplateTemplateParm, Index,
            ((PackIndex.toInternalRepresentation()) << 1) | Final),
        Replacement(Replacement), AssociatedDecl(AssociatedDecl) {
    assert(AssociatedDecl != nullptr);
  }

public:
  /// A template-like entity which owns the whole pattern being substituted.
  /// This will own a set of template parameters.
  Decl *getAssociatedDecl() const { return AssociatedDecl; }

  /// Returns the index of the replaced parameter in the associated declaration.
  /// This should match the result of `getParameter()->getIndex()`.
  unsigned getIndex() const { return Bits.Index; }

  // This substitution is Final, which means the substitution is fully
  // sugared: it doesn't need to be resugared later.
  bool getFinal() const { return Bits.Data & 1; }

  UnsignedOrNone getPackIndex() const {
    return UnsignedOrNone::fromInternalRepresentation(Bits.Data >> 1);
  }

  TemplateTemplateParmDecl *getParameter() const;
  TemplateName getReplacement() const { return Replacement; }

  void Profile(llvm::FoldingSetNodeID &ID);

  static void Profile(llvm::FoldingSetNodeID &ID, TemplateName Replacement,
                      Decl *AssociatedDecl, unsigned Index,
                      UnsignedOrNone PackIndex, bool Final);
};

class DeducedTemplateStorage : public UncommonTemplateNameStorage,
                               public llvm::FoldingSetNode {
  friend class ASTContext;

  TemplateName Underlying;

  DeducedTemplateStorage(TemplateName Underlying,
                         const DefaultArguments &DefArgs);

public:
  TemplateName getUnderlying() const { return Underlying; }

  DefaultArguments getDefaultArguments() const {
    return {/*StartPos=*/Bits.Index,
            /*Args=*/{reinterpret_cast<const TemplateArgument *>(this + 1),
                      Bits.Data}};
  }

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context) const;

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      TemplateName Underlying, const DefaultArguments &DefArgs);
};

inline TemplateName TemplateName::getUnderlying() const {
  if (SubstTemplateTemplateParmStorage *subst
        = getAsSubstTemplateTemplateParm())
    return subst->getReplacement().getUnderlying();
  return *this;
}

/// Represents a template name as written in source code.
///
/// This kind of template name may refer to a template name that was
/// preceded by a nested name specifier, e.g., \c std::vector. Here,
/// the nested name specifier is "std::" and the template name is the
/// declaration for "vector". It may also have been written with the
/// 'template' keyword. The QualifiedTemplateName class is only
/// used to provide "sugar" for template names, so that they can
/// be differentiated from canonical template names. and has no
/// semantic meaning. In this manner, it is to TemplateName what
/// ElaboratedType is to Type, providing extra syntactic sugar
/// for downstream clients.
class QualifiedTemplateName : public llvm::FoldingSetNode {
  friend class ASTContext;

  /// The nested name specifier that qualifies the template name.
  ///
  /// The bit is used to indicate whether the "template" keyword was
  /// present before the template name itself. Note that the
  /// "template" keyword is always redundant in this case (otherwise,
  /// the template name would be a dependent name and we would express
  /// this name with DependentTemplateName).
  llvm::PointerIntPair<NestedNameSpecifier, 1, bool> Qualifier;

  /// The underlying template name, it is either
  ///  1) a Template -- a template declaration that this qualified name refers
  ///     to.
  ///  2) or a UsingTemplate -- a template declaration introduced by a
  ///     using-shadow declaration.
  TemplateName UnderlyingTemplate;

  QualifiedTemplateName(NestedNameSpecifier NNS, bool TemplateKeyword,
                        TemplateName Template)
      : Qualifier(NNS, TemplateKeyword ? 1 : 0), UnderlyingTemplate(Template) {
    assert(UnderlyingTemplate.getKind() == TemplateName::Template ||
           UnderlyingTemplate.getKind() == TemplateName::UsingTemplate);
  }

public:
  /// Return the nested name specifier that qualifies this name.
  NestedNameSpecifier getQualifier() const { return Qualifier.getPointer(); }

  /// Whether the template name was prefixed by the "template"
  /// keyword.
  bool hasTemplateKeyword() const { return Qualifier.getInt(); }

  /// Return the underlying template name.
  TemplateName getUnderlyingTemplate() const { return UnderlyingTemplate; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getQualifier(), hasTemplateKeyword(), UnderlyingTemplate);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier NNS,
                      bool TemplateKeyword, TemplateName TN) {
    NNS.Profile(ID);
    ID.AddBoolean(TemplateKeyword);
    ID.AddPointer(TN.getAsVoidPointer());
  }
};

struct IdentifierOrOverloadedOperator {
  IdentifierOrOverloadedOperator() = default;
  IdentifierOrOverloadedOperator(const IdentifierInfo *II);
  IdentifierOrOverloadedOperator(OverloadedOperatorKind OOK);

  /// Returns the identifier to which this template name refers.
  const IdentifierInfo *getIdentifier() const {
    if (getOperator() != OO_None)
      return nullptr;
    return reinterpret_cast<const IdentifierInfo *>(PtrOrOp);
  }

  /// Return the overloaded operator to which this template name refers.
  OverloadedOperatorKind getOperator() const {
    uintptr_t OOK = -PtrOrOp;
    return OOK < NUM_OVERLOADED_OPERATORS ? OverloadedOperatorKind(OOK)
                                          : OO_None;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const;

  bool operator==(const IdentifierOrOverloadedOperator &Other) const {
    return PtrOrOp == Other.PtrOrOp;
  };

private:
  uintptr_t PtrOrOp = 0;
};

/// Represents a dependent template name that cannot be
/// resolved prior to template instantiation.
///
/// This kind of template name refers to a dependent template name,
/// including its nested name specifier (if any). For example,
/// DependentTemplateName can refer to "MetaFun::template apply",
/// where "MetaFun::" is the nested name specifier and "apply" is the
/// template name referenced. The "template" keyword is implied.
class DependentTemplateStorage {
  /// The nested name specifier that qualifies the template
  /// name.
  ///
  /// The bit stored in this qualifier describes whether the \c Name field
  /// was preceeded by a template keyword.
  llvm::PointerIntPair<NestedNameSpecifier, 1, bool> Qualifier;

  /// The dependent template name.
  IdentifierOrOverloadedOperator Name;

public:
  DependentTemplateStorage(NestedNameSpecifier Qualifier,
                           IdentifierOrOverloadedOperator Name,
                           bool HasTemplateKeyword);

  /// Return the nested name specifier that qualifies this name.
  NestedNameSpecifier getQualifier() const { return Qualifier.getPointer(); }

  IdentifierOrOverloadedOperator getName() const { return Name; }

  /// Was this template name was preceeded by the template keyword?
  bool hasTemplateKeyword() const { return Qualifier.getInt(); }

  TemplateNameDependence getDependence() const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, getQualifier(), getName(), hasTemplateKeyword());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier NNS,
                      IdentifierOrOverloadedOperator Name,
                      bool HasTemplateKeyword) {
    NNS.Profile(ID);
    ID.AddBoolean(HasTemplateKeyword);
    Name.Profile(ID);
  }

  void print(raw_ostream &OS, const PrintingPolicy &Policy) const;
};

class DependentTemplateName : public DependentTemplateStorage,
                              public llvm::FoldingSetNode {
  friend class ASTContext;
  using DependentTemplateStorage::DependentTemplateStorage;
  DependentTemplateName(const DependentTemplateStorage &S)
      : DependentTemplateStorage(S) {}
};

} // namespace clang.

namespace llvm {

/// The clang::TemplateName class is effectively a pointer.
template<>
struct PointerLikeTypeTraits<clang::TemplateName> {
  static inline void *getAsVoidPointer(clang::TemplateName TN) {
    return TN.getAsVoidPointer();
  }

  static inline clang::TemplateName getFromVoidPointer(void *Ptr) {
    return clang::TemplateName::getFromVoidPointer(Ptr);
  }

  // No bits are available!
  static constexpr int NumLowBitsAvailable = 0;
};

} // namespace llvm.

#endif // LLVM_CLANG_AST_TEMPLATENAME_H
