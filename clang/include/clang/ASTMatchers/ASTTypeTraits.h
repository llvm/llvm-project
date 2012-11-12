//===--- ASTMatchersTypeTraits.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides a dynamically typed node container that can be used to store
//  an AST base node at runtime in the same storage in a type safe way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H
#define LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/AlignOf.h"

namespace clang {
namespace ast_type_traits {

/// \brief A dynamically typed AST node container.
///
/// Stores an AST node in a type safe way. This allows writing code that
/// works with different kinds of AST nodes, despite the fact that they don't
/// have a common base class.
///
/// Use \c create(Node) to create a \c DynTypedNode from an AST node,
/// and \c get<T>() to retrieve the node as type T if the types match.
///
/// See \c NodeTypeTag for which node base types are currently supported;
/// You can create DynTypedNodes for all nodes in the inheritance hierarchy of
/// the supported base types.
class DynTypedNode {
public:
  /// \brief Creates a \c DynTypedNode from \c Node.
  template <typename T>
  static DynTypedNode create(const T &Node) {
    return BaseConverter<T>::create(Node);
  }

  /// \brief Retrieve the stored node as type \c T.
  ///
  /// Returns NULL if the stored node does not have a type that is
  /// convertible to \c T.
  ///
  /// For types that have identity via their pointer in the AST
  /// (like \c Stmt and \c Decl) the returned pointer points to the
  /// referenced AST node.
  /// For other types (like \c QualType) the value is stored directly
  /// in the \c DynTypedNode, and the returned pointer points at
  /// the storage inside DynTypedNode. For those nodes, do not
  /// use the pointer outside the scope of the DynTypedNode.
  template <typename T>
  const T *get() const {
    return BaseConverter<T>::get(Tag, Storage.buffer);
  }

  /// \brief Returns a pointer that identifies the stored AST node.
  ///
  /// Note that this is not supported by all AST nodes. For AST nodes
  /// that don't have a pointer-defined identity inside the AST, this
  /// method returns NULL.
  const void *getMemoizationData() const;

private:
  /// \brief Takes care of converting from and to \c T.
  template <typename T, typename EnablerT = void> struct BaseConverter;

  /// \brief Supported base node types.
  enum NodeTypeTag {
    NT_Decl,
    NT_Stmt,
    NT_NestedNameSpecifier,
    NT_NestedNameSpecifierLoc,
    NT_QualType,
    NT_Type,
    NT_TypeLoc
  } Tag;

  /// \brief Stores the data of the node.
  ///
  /// Note that we can store \c Decls and \c Stmts by pointer as they are
  /// guaranteed to be unique pointers pointing to dedicated storage in the
  /// AST. \c QualTypes on the other hand do not have storage or unique
  /// pointers and thus need to be stored by value.
  llvm::AlignedCharArrayUnion<Decl*, QualType, TypeLoc, NestedNameSpecifierLoc>
    Storage;
};

// FIXME: Pull out abstraction for the following.
template<typename T> struct DynTypedNode::BaseConverter<T,
    typename llvm::enable_if<llvm::is_base_of<Decl, T> >::type> {
  static const T *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_Decl)
      return dyn_cast<T>(*reinterpret_cast<Decl*const*>(Storage));
    return NULL;
  }
  static DynTypedNode create(const Decl &Node) {
    DynTypedNode Result;
    Result.Tag = NT_Decl;
    new (Result.Storage.buffer) const Decl*(&Node);
    return Result;
  }
};
template<typename T> struct DynTypedNode::BaseConverter<T,
    typename llvm::enable_if<llvm::is_base_of<Stmt, T> >::type> {
  static const T *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_Stmt)
      return dyn_cast<T>(*reinterpret_cast<Stmt*const*>(Storage));
    return NULL;
  }
  static DynTypedNode create(const Stmt &Node) {
    DynTypedNode Result;
    Result.Tag = NT_Stmt;
    new (Result.Storage.buffer) const Stmt*(&Node);
    return Result;
  }
};
template<typename T> struct DynTypedNode::BaseConverter<T,
    typename llvm::enable_if<llvm::is_base_of<Type, T> >::type> {
  static const T *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_Type)
      return dyn_cast<T>(*reinterpret_cast<Type*const*>(Storage));
    return NULL;
  }
  static DynTypedNode create(const Type &Node) {
    DynTypedNode Result;
    Result.Tag = NT_Type;
    new (Result.Storage.buffer) const Type*(&Node);
    return Result;
  }
};
template<> struct DynTypedNode::BaseConverter<NestedNameSpecifier, void> {
  static const NestedNameSpecifier *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_NestedNameSpecifier)
      return *reinterpret_cast<NestedNameSpecifier*const*>(Storage);
    return NULL;
  }
  static DynTypedNode create(const NestedNameSpecifier &Node) {
    DynTypedNode Result;
    Result.Tag = NT_NestedNameSpecifier;
    new (Result.Storage.buffer) const NestedNameSpecifier*(&Node);
    return Result;
  }
};
template<> struct DynTypedNode::BaseConverter<NestedNameSpecifierLoc, void> {
  static const NestedNameSpecifierLoc *get(NodeTypeTag Tag,
                                           const char Storage[]) {
    if (Tag == NT_NestedNameSpecifierLoc)
      return reinterpret_cast<const NestedNameSpecifierLoc*>(Storage);
    return NULL;
  }
  static DynTypedNode create(const NestedNameSpecifierLoc &Node) {
    DynTypedNode Result;
    Result.Tag = NT_NestedNameSpecifierLoc;
    new (Result.Storage.buffer) NestedNameSpecifierLoc(Node);
    return Result;
  }
};
template<> struct DynTypedNode::BaseConverter<QualType, void> {
  static const QualType *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_QualType)
      return reinterpret_cast<const QualType*>(Storage);
    return NULL;
  }
  static DynTypedNode create(const QualType &Node) {
    DynTypedNode Result;
    Result.Tag = NT_QualType;
    new (Result.Storage.buffer) QualType(Node);
    return Result;
  }
};
template<> struct DynTypedNode::BaseConverter<TypeLoc, void> {
  static const TypeLoc *get(NodeTypeTag Tag, const char Storage[]) {
    if (Tag == NT_TypeLoc)
      return reinterpret_cast<const TypeLoc*>(Storage);
    return NULL;
  }
  static DynTypedNode create(const TypeLoc &Node) {
    DynTypedNode Result;
    Result.Tag = NT_TypeLoc;
    new (Result.Storage.buffer) TypeLoc(Node);
    return Result;
  }
};
// The only operation we allow on unsupported types is \c get.
// This allows to conveniently use \c DynTypedNode when having an arbitrary
// AST node that is not supported, but prevents misuse - a user cannot create
// a DynTypedNode from arbitrary types.
template <typename T, typename EnablerT> struct DynTypedNode::BaseConverter {
  static const T *get(NodeTypeTag Tag, const char Storage[]) { return NULL; }
};

inline const void *DynTypedNode::getMemoizationData() const {
  switch (Tag) {
    case NT_Decl: return BaseConverter<Decl>::get(Tag, Storage.buffer);
    case NT_Stmt: return BaseConverter<Stmt>::get(Tag, Storage.buffer);
    default: return NULL;
  };
}

} // end namespace ast_type_traits
} // end namespace clang

#endif // LLVM_CLANG_AST_MATCHERS_AST_TYPE_TRAITS_H
