//===--- DeclRefExprUtils.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeclRefExprUtils.h"
#include "Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <cassert>

namespace clang::tidy::utils::decl_ref_expr {

using namespace ::clang::ast_matchers;
using llvm::SmallPtrSet;

namespace {

template <typename S> bool isSetDifferenceEmpty(const S &S1, const S &S2) {
  for (auto E : S1)
    if (S2.count(E) == 0)
      return false;
  return true;
}

// Extracts all Nodes keyed by ID from Matches and inserts them into Nodes.
template <typename Node>
void extractNodesByIdTo(ArrayRef<BoundNodes> Matches, StringRef ID,
                        SmallPtrSet<const Node *, 16> &Nodes) {
  for (const auto &Match : Matches)
    Nodes.insert(Match.getNodeAs<Node>(ID));
}

// Returns true if both types refer to the same type,
// ignoring the const-qualifier.
bool isSameTypeIgnoringConst(QualType A, QualType B) {
  A = A.getCanonicalType();
  B = B.getCanonicalType();
  A.addConst();
  B.addConst();
  return A == B;
}

// Returns true if `D` and `O` have the same parameter types.
bool hasSameParameterTypes(const CXXMethodDecl &D, const CXXMethodDecl &O) {
  if (D.getNumParams() != O.getNumParams())
    return false;
  for (int I = 0, E = D.getNumParams(); I < E; ++I) {
    if (!isSameTypeIgnoringConst(D.getParamDecl(I)->getType(),
                                 O.getParamDecl(I)->getType()))
      return false;
  }
  return true;
}

// If `D` has a const-qualified overload with otherwise identical
// ref-qualifiers and parameter types, returns that overload.
const CXXMethodDecl *findConstOverload(const CXXMethodDecl &D) {
  assert(!D.isConst());

  DeclContext::lookup_result LookupResult =
      D.getParent()->lookup(D.getNameInfo().getName());
  if (LookupResult.isSingleResult()) {
    // No overload.
    return nullptr;
  }
  for (const Decl *Overload : LookupResult) {
    const auto *O = dyn_cast<CXXMethodDecl>(Overload);
    if (O && !O->isDeleted() && O->isConst() &&
        O->getRefQualifier() == D.getRefQualifier() &&
        hasSameParameterTypes(D, *O))
      return O;
  }
  return nullptr;
}

// Returns true if both types are pointers or reference to the same type,
// ignoring the const-qualifier.
bool pointsToSameTypeIgnoringConst(QualType A, QualType B) {
  assert(A->isPointerType() || A->isReferenceType());
  assert(B->isPointerType() || B->isReferenceType());
  return isSameTypeIgnoringConst(A->getPointeeType(), B->getPointeeType());
}

// Return true if non-const member function `M` likely does not mutate `*this`.
//
// Note that if the member call selects a method/operator `f` that
// is not const-qualified, then we also consider that the object is
// not mutated if:
//  - (A) there is a const-qualified overload `cf` of `f` that has
//  the
//    same ref-qualifiers;
//  - (B) * `f` returns a value, or
//        * if `f` returns a `T&`, `cf` returns a `const T&` (up to
//          possible aliases such as `reference` and
//          `const_reference`), or
//        * if `f` returns a `T*`, `cf` returns a `const T*` (up to
//          possible aliases).
//  - (C) the result of the call is not mutated.
//
// The assumption that `cf` has the same semantics as `f`.
// For example:
//   - In `std::vector<T> v; const T t = v[...];`, we consider that
//     expression `v[...]` does not mutate `v` as
//    `T& std::vector<T>::operator[]` has a const overload
//     `const T& std::vector<T>::operator[] const`, and the
//     result expression of type `T&` is only used as a `const T&`;
//   - In `std::map<K, V> m; V v = m.at(...);`, we consider
//     `m.at(...)` to be an immutable access for the same reason.
// However:
//   - In `std::map<K, V> m; const V v = m[...];`, We consider that
//     `m[...]` mutates `m` as `V& std::map<K, V>::operator[]` does
//     not have a const overload.
//   - In `std::vector<T> v; T& t = v[...];`, we consider that
//     expression `v[...]` mutates `v` as the result is kept as a
//     mutable reference.
//
// This function checks (A) ad (B), but the caller should make sure that the
// object is not mutated through the return value.
bool isLikelyShallowConst(const CXXMethodDecl &M) {
  assert(!M.isConst());
  // The method can mutate our variable.

  // (A)
  const CXXMethodDecl *ConstOverload = findConstOverload(M);
  if (ConstOverload == nullptr) {
    return false;
  }

  // (B)
  const QualType CallTy = M.getReturnType().getCanonicalType();
  const QualType OverloadTy = ConstOverload->getReturnType().getCanonicalType();
  if (CallTy->isReferenceType()) {
    return OverloadTy->isReferenceType() &&
           pointsToSameTypeIgnoringConst(CallTy, OverloadTy);
  }
  if (CallTy->isPointerType()) {
    return OverloadTy->isPointerType() &&
           pointsToSameTypeIgnoringConst(CallTy, OverloadTy);
  }
  return isSameTypeIgnoringConst(CallTy, OverloadTy);
}

// A matcher that matches DeclRefExprs that are used in ways such that the
// underlying declaration is not modified.
// If the declaration is of pointer type, `Indirections` specifies the level
// of indirection of the object whose mutations we are tracking.
//
// For example, given:
//   ```
//   int i;
//   int* p;
//   p = &i;  // (A)
//   *p = 3;  // (B)
//   ```
//
//  `declRefExpr(to(varDecl(hasName("p"))), doesNotMutateObject(0))` matches
//  (B), but `declRefExpr(to(varDecl(hasName("p"))), doesNotMutateObject(1))`
//  matches (A).
//
AST_MATCHER_P(DeclRefExpr, doesNotMutateObject, int, Indirections) {
  // We walk up the parents of the DeclRefExpr recursively. There are a few
  // kinds of expressions:
  //  - Those that cannot be used to mutate the underlying variable. We can stop
  //    recursion there.
  //  - Those that can be used to mutate the underlying variable in analyzable
  //    ways (such as taking the address or accessing a subobject). We have to
  //    examine the parents.
  //  - Those that we don't know how to analyze. In that case we stop there and
  //    we assume that they can modify the expression.

  struct StackEntry {
    StackEntry(const Expr *E, int Indirections)
        : E(E), Indirections(Indirections) {}
    // The expression to analyze.
    const Expr *E;
    // The number of pointer indirections of the object being tracked (how
    // many times an address was taken).
    int Indirections;
  };

  llvm::SmallVector<StackEntry, 4> Stack;
  Stack.emplace_back(&Node, Indirections);
  ASTContext &Ctx = Finder->getASTContext();

  while (!Stack.empty()) {
    const StackEntry Entry = Stack.back();
    Stack.pop_back();

    // If the expression type is const-qualified at the appropriate indirection
    // level then we can not mutate the object.
    QualType Ty = Entry.E->getType().getCanonicalType();
    for (int I = 0; I < Entry.Indirections; ++I) {
      assert(Ty->isPointerType());
      Ty = Ty->getPointeeType().getCanonicalType();
    }
    if (Ty->isVoidType() || Ty.isConstQualified())
      continue;

    // Otherwise we have to look at the parents to see how the expression is
    // used.
    const DynTypedNodeList Parents = Ctx.getParents(*Entry.E);
    // Note: most nodes have a single parents, but there exist nodes that have
    // several parents, such as `InitListExpr` that have semantic and syntactic
    // forms.
    for (const auto &Parent : Parents) {
      if (Parent.get<CompoundStmt>()) {
        // Unused block-scope statement.
        continue;
      }
      const Expr *const P = Parent.get<Expr>();
      if (P == nullptr) {
        // `Parent` is not an expr (e.g. a `VarDecl`).
        // The case of binding to a `const&` or `const*` variable is handled by
        // the fact that there is going to be a `NoOp` cast to const below the
        // `VarDecl`, so we're not even going to get there.
        // The case of copying into a value-typed variable is handled by the
        // rvalue cast.
        // This triggers only when binding to a mutable reference/ptr variable.
        // FIXME: When we take a mutable reference we could keep checking the
        // new variable for const usage only.
        return false;
      }
      // Cosmetic nodes.
      if (isa<ParenExpr>(P) || isa<MaterializeTemporaryExpr>(P)) {
        Stack.emplace_back(P, Entry.Indirections);
        continue;
      }
      if (const auto *const Cast = dyn_cast<CastExpr>(P)) {
        switch (Cast->getCastKind()) {
        // NoOp casts are used to add `const`. We'll check whether adding that
        // const prevents modification when we process the cast.
        case CK_NoOp:
        // These do nothing w.r.t. to mutability.
        case CK_BaseToDerived:
        case CK_DerivedToBase:
        case CK_UncheckedDerivedToBase:
        case CK_Dynamic:
        case CK_BaseToDerivedMemberPointer:
        case CK_DerivedToBaseMemberPointer:
          Stack.emplace_back(Cast, Entry.Indirections);
          continue;
        case CK_ToVoid:
        case CK_PointerToBoolean:
          // These do not mutate the underlying variable.
          continue;
        case CK_LValueToRValue: {
          // An rvalue is immutable.
          if (Entry.Indirections == 0)
            continue;
          Stack.emplace_back(Cast, Entry.Indirections);
          continue;
        }
        default:
          // Bail out on casts that we cannot analyze.
          return false;
        }
      }
      if (const auto *const Member = dyn_cast<MemberExpr>(P)) {
        if (const auto *const Method =
                dyn_cast<CXXMethodDecl>(Member->getMemberDecl())) {
          if (Method->isConst() || Method->isStatic()) {
            // The method call cannot mutate our variable.
            continue;
          }
          if (isLikelyShallowConst(*Method)) {
            // We still have to check that the object is not modified through
            // the method's return value (C).
            const auto MemberParents = Ctx.getParents(*Member);
            assert(MemberParents.size() == 1);
            const auto *Call = MemberParents[0].get<CallExpr>();
            // If `o` is an object of class type and `f` is a member function,
            // then `o.f` has to be used as part of a call expression.
            assert(Call != nullptr && "member function has to be called");
            Stack.emplace_back(
                Call,
                Method->getReturnType().getCanonicalType()->isPointerType()
                    ? 1
                    : 0);
            continue;
          }
          return false;
        }
        Stack.emplace_back(Member, 0);
        continue;
      }
      if (const auto *const OpCall = dyn_cast<CXXOperatorCallExpr>(P)) {
        // Operator calls have function call syntax. The `*this` parameter
        // is the first parameter.
        if (OpCall->getNumArgs() == 0 || OpCall->getArg(0) != Entry.E) {
          return false;
        }
        const auto *const Method =
            dyn_cast_or_null<CXXMethodDecl>(OpCall->getDirectCallee());

        if (Method == nullptr) {
          // This is not a member operator. Typically, a friend operator. These
          // are handled like function calls.
          return false;
        }

        if (Method->isConst() || Method->isStatic()) {
          continue;
        }
        if (isLikelyShallowConst(*Method)) {
          // We still have to check that the object is not modified through
          // the operator's return value (C).
          Stack.emplace_back(
              OpCall,
              Method->getReturnType().getCanonicalType()->isPointerType() ? 1
                                                                          : 0);
          continue;
        }
        return false;
      }

      if (const auto *const Op = dyn_cast<UnaryOperator>(P)) {
        switch (Op->getOpcode()) {
        case UO_AddrOf:
          Stack.emplace_back(Op, Entry.Indirections + 1);
          continue;
        case UO_Deref:
          assert(Entry.Indirections > 0);
          Stack.emplace_back(Op, Entry.Indirections - 1);
          continue;
        default:
          // Bail out on unary operators that we cannot analyze.
          return false;
        }
      }

      // Assume any other expression can modify the underlying variable.
      return false;
    }
  }

  // No parent can modify the variable.
  return true;
}

} // namespace

SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt,
                           ASTContext &Context, int Indirections) {
  auto Matches = match(findAll(declRefExpr(to(varDecl(equalsNode(&VarDecl))),
                                           doesNotMutateObject(Indirections))
                                   .bind("declRef")),
                       Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);

  return DeclRefs;
}

bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context, int Indirections) {
  // Collect all DeclRefExprs to the loop variable and all CallExprs and
  // CXXConstructExprs where the loop variable is used as argument to a const
  // reference parameter.
  // If the difference is empty it is safe for the loop variable to be a const
  // reference.
  auto AllDeclRefs = allDeclRefExprs(Var, Stmt, Context);
  auto ConstReferenceDeclRefs =
      constReferenceDeclRefExprs(Var, Stmt, Context, Indirections);
  return isSetDifferenceEmpty(AllDeclRefs, ConstReferenceDeclRefs);
}

SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context) {
  auto Matches = match(
      findAll(declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef")),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl, ASTContext &Context) {
  auto Matches = match(
      decl(forEachDescendant(
          declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef"))),
      Decl, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

bool isCopyConstructorArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                               ASTContext &Context) {
  auto UsedAsConstRefArg = forEachArgumentWithParam(
      declRefExpr(equalsNode(&DeclRef)),
      parmVarDecl(hasType(matchers::isReferenceToConst())));
  auto Matches = match(
      decl(hasDescendant(
          cxxConstructExpr(UsedAsConstRefArg, hasDeclaration(cxxConstructorDecl(
                                                  isCopyConstructor())))
              .bind("constructExpr"))),
      Decl, Context);
  return !Matches.empty();
}

bool isCopyAssignmentArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                              ASTContext &Context) {
  auto UsedAsConstRefArg = forEachArgumentWithParam(
      declRefExpr(equalsNode(&DeclRef)),
      parmVarDecl(hasType(matchers::isReferenceToConst())));
  auto Matches = match(
      decl(hasDescendant(
          cxxOperatorCallExpr(UsedAsConstRefArg, hasOverloadedOperatorName("="),
                              callee(cxxMethodDecl(isCopyAssignmentOperator())))
              .bind("operatorCallExpr"))),
      Decl, Context);
  return !Matches.empty();
}

} // namespace clang::tidy::utils::decl_ref_expr
