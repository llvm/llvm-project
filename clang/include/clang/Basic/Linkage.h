//===--- Linkage.h - Linkage enumeration and utilities ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the Linkage enumeration and various utility functions.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_LINKAGE_H
#define LLVM_CLANG_BASIC_LINKAGE_H

namespace clang {

/// \brief Describes the different kinds of linkage 
/// (C++ [basic.link], C99 6.2.2) that an entity may have.
enum Linkage {
  /// \brief No linkage, which means that the entity is unique and
  /// can only be referred to from within its scope.
  NoLinkage = 0,

  /// \brief Internal linkage, which indicates that the entity can
  /// be referred to from within the translation unit (but not other
  /// translation units).
  InternalLinkage,

  /// \brief External linkage within a unique namespace. 
  ///
  /// From the language perspective, these entities have external
  /// linkage. However, since they reside in an anonymous namespace,
  /// their names are unique to this translation unit, which is
  /// equivalent to having internal linkage from the code-generation
  /// point of view.
  UniqueExternalLinkage,

  /// \brief External linkage, which indicates that the entity can
  /// be referred to from other translation units.
  ExternalLinkage
};

/// \brief A more specific kind of linkage than enum Linkage.
///
/// This is relevant to CodeGen and AST file reading.
enum GVALinkage {
  GVA_Internal,
  GVA_C99Inline,
  GVA_CXXInline,
  GVA_StrongExternal,
  GVA_TemplateInstantiation,
  GVA_ExplicitTemplateInstantiation
};

/// \brief Determine whether the given linkage is semantically external.
inline bool isExternalLinkage(Linkage L) {
  return L == UniqueExternalLinkage || L == ExternalLinkage;
}

/// \brief Compute the minimum linkage given two linages.
inline Linkage minLinkage(Linkage L1, Linkage L2) {
  return L1 < L2? L1 : L2;
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_LINKAGE_H
