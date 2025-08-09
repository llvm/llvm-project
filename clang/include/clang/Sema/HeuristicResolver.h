//===--- HeuristicResolver.h - Resolution of dependent names -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_HEURISTICRESOLVER_H
#define LLVM_CLANG_SEMA_HEURISTICRESOLVER_H

#include "clang/AST/Decl.h"
#include <vector>

namespace clang {

class ASTContext;
class CallExpr;
class CXXBasePath;
class CXXDependentScopeMemberExpr;
class DeclarationName;
class DependentScopeDeclRefExpr;
class FunctionProtoTypeLoc;
class NamedDecl;
class Type;
class UnresolvedUsingValueDecl;

// This class handles heuristic resolution of declarations and types in template
// code.
//
// As a compiler, clang only needs to perform certain types of processing on
// template code (such as resolving dependent names to declarations, or
// resolving the type of a dependent expression) after instantiation. Indeed,
// C++ language features such as template specialization mean such resolution
// cannot be done accurately before instantiation.
//
// However, template code is written and read in uninstantiated form, and clangd
// would like to provide editor features like go-to-definition in template code
// where possible. To this end, clangd attempts to resolve declarations and
// types in uninstantiated code by using heuristics, understanding that the
// results may not be fully accurate but that this is better than nothing.
//
// At this time, the heuristic used is a simple but effective one: assume that
// template instantiations are based on the primary template definition and not
// not a specialization. More advanced heuristics may be added in the future.
class HeuristicResolver {
public:
  HeuristicResolver(ASTContext &Ctx) : Ctx(Ctx) {}

  // Try to heuristically resolve certain types of expressions, declarations, or
  // types to one or more likely-referenced declarations.
  std::vector<const NamedDecl *>
  resolveMemberExpr(const CXXDependentScopeMemberExpr *ME) const;
  std::vector<const NamedDecl *>
  resolveDeclRefExpr(const DependentScopeDeclRefExpr *RE) const;
  std::vector<const NamedDecl *>
  resolveTypeOfCallExpr(const CallExpr *CE) const;
  std::vector<const NamedDecl *>
  resolveCalleeOfCallExpr(const CallExpr *CE) const;
  std::vector<const NamedDecl *>
  resolveUsingValueDecl(const UnresolvedUsingValueDecl *UUVD) const;
  std::vector<const NamedDecl *>
  resolveDependentNameType(const DependentNameType *DNT) const;
  std::vector<const NamedDecl *> resolveTemplateSpecializationType(
      const DependentTemplateSpecializationType *DTST) const;

  // Try to heuristically resolve a dependent nested name specifier
  // to the type it likely denotes. Note that *dependent* name specifiers always
  // denote types, not namespaces.
  QualType resolveNestedNameSpecifierToType(NestedNameSpecifier NNS) const;

  // Perform an imprecise lookup of a dependent name in `RD`.
  // This function does not follow strict semantic rules and should be used
  // only when lookup rules can be relaxed, e.g. indexing.
  std::vector<const NamedDecl *>
  lookupDependentName(CXXRecordDecl *RD, DeclarationName Name,
                      llvm::function_ref<bool(const NamedDecl *ND)> Filter);

  // Given the type T of a dependent expression that appears of the LHS of a
  // "->", heuristically find a corresponding pointee type in whose scope we
  // could look up the name appearing on the RHS.
  const QualType getPointeeType(QualType T) const;

  // Heuristically resolve a possibly-dependent type `T` to a TagDecl
  // in which a member's name can be looked up.
  TagDecl *resolveTypeToTagDecl(QualType T) const;

  // Simplify the type `Type`.
  // `E` is the expression whose type `Type` is, if known. This sometimes
  // contains information relevant to the type that's not stored in `Type`
  // itself.
  // If `UnwrapPointer` is true, exactly only pointer type will be unwrapped
  // during simplification, and the operation fails if no pointer type is found.
  QualType simplifyType(QualType Type, const Expr *E, bool UnwrapPointer);

  // Given an expression `Fn` representing the callee in a function call,
  // if the call is through a function pointer, try to find the declaration of
  // the corresponding function pointer type, so that we can recover argument
  // names from it.
  FunctionProtoTypeLoc getFunctionProtoTypeLoc(const Expr *Fn) const;

private:
  ASTContext &Ctx;
};

} // namespace clang

#endif
