//===- ASTCommon.h - Common stuff for ASTReader/ASTWriter -*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SERIALIZATION_ASTCOMMON_H
#define LLVM_CLANG_LIB_SERIALIZATION_ASTCOMMON_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/Basic/LLVM.h"
#include "clang/Serialization/ASTBitCodes.h"

namespace clang {

namespace serialization {

enum class DeclUpdateKind {
  CXXAddedImplicitMember,
  CXXAddedAnonymousNamespace,
  CXXAddedFunctionDefinition,
  CXXAddedVarDefinition,
  CXXPointOfInstantiation,
  CXXInstantiatedClassDefinition,
  CXXInstantiatedDefaultArgument,
  CXXInstantiatedDefaultMemberInitializer,
  CXXResolvedDtorDelete,
  CXXResolvedExceptionSpec,
  CXXDeducedReturnType,
  DeclMarkedUsed,
  ManglingNumber,
  StaticLocalNumber,
  DeclMarkedOpenMPThreadPrivate,
  DeclMarkedOpenMPAllocate,
  DeclMarkedOpenMPDeclareTarget,
  DeclExported,
  AddedAttrToRecord,
  CXXResolvedDtorGlobDelete,
  CXXResolvedDtorArrayDelete,
  CXXResolvedDtorGlobArrayDelete
};

TypeIdx TypeIdxFromBuiltin(const BuiltinType *BT);

unsigned ComputeHash(Selector Sel);

/// Retrieve the "definitive" declaration that provides all of the
/// visible entries for the given declaration context, if there is one.
///
/// The "definitive" declaration is the only place where we need to look to
/// find information about the declarations within the given declaration
/// context. For example, C++ and Objective-C classes, C structs/unions, and
/// Objective-C protocols, categories, and extensions are all defined in a
/// single place in the source code, so they have definitive declarations
/// associated with them. C++ namespaces, on the other hand, can have
/// multiple definitions.
const DeclContext *getDefinitiveDeclContext(const DeclContext *DC);

/// Determine whether the given declaration kind is redeclarable.
bool isRedeclarableDeclKind(unsigned Kind);

/// Determine whether the given declaration needs an anonymous
/// declaration number.
bool needsAnonymousDeclarationNumber(const NamedDecl *D);

/// Visit each declaration within \c DC that needs an anonymous
/// declaration number and call \p Visit with the declaration and its number.
template<typename Fn> void numberAnonymousDeclsWithin(const DeclContext *DC,
                                                      Fn Visit) {
  unsigned Index = 0;
  for (Decl *LexicalD : DC->decls()) {
    // For a friend decl, we care about the declaration within it, if any.
    if (auto *FD = dyn_cast<FriendDecl>(LexicalD))
      LexicalD = FD->getFriendDecl();

    auto *ND = dyn_cast_or_null<NamedDecl>(LexicalD);
    if (!ND || !needsAnonymousDeclarationNumber(ND))
      continue;

    Visit(ND, Index++);
  }
}

/// Determine whether the given declaration will be included in the per-module
/// initializer if it needs to be eagerly handed to the AST consumer. If so, we
/// should not hand it to the consumer when deserializing it, nor include it in
/// the list of eagerly deserialized declarations.
inline bool isPartOfPerModuleInitializer(const Decl *D) {
  if (isa<ImportDecl>(D))
    return true;
  // Template instantiations are notionally in an "instantiation unit" rather
  // than in any particular translation unit, so they need not be part of any
  // particular (sub)module's per-module initializer.
  if (auto *VD = dyn_cast<VarDecl>(D))
    return !isTemplateInstantiation(VD->getTemplateSpecializationKind());
  return false;
}

} // namespace serialization

} // namespace clang

#endif
