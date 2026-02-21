//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSpanParamCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

// Methods on std::vector that only read data (compatible with std::span).
static bool isReadOnlyVectorMethod(StringRef Name) {
  return Name == "operator[]" || Name == "at" || Name == "data" ||
         Name == "size" || Name == "empty" || Name == "begin" ||
         Name == "end" || Name == "cbegin" || Name == "cend" ||
         Name == "rbegin" || Name == "rend" || Name == "crbegin" ||
         Name == "crend" || Name == "front" || Name == "back";
}

// Check if all uses of the parameter in the function body are read-only.
static bool allUsesAreReadOnly(const ParmVarDecl *Param,
                               const FunctionDecl *Func, ASTContext &Context) {
  const Stmt *Body = Func->getBody();
  if (!Body)
    return false;

  const auto Refs = match(
      findAll(declRefExpr(to(equalsNode(Param))).bind("ref")), *Body, Context);

  for (const auto &Ref : Refs) {
    const auto *DRE = Ref.getNodeAs<DeclRefExpr>("ref");
    if (!DRE)
      return false;

    // Walk up through implicit casts to find the "real" parent.
    const Expr *Current = DRE;
    while (true) {
      const auto Parents = Context.getParents(*Current);
      if (Parents.empty())
        return false;
      const auto &Parent = Parents[0];
      if (const auto *ICE = Parent.get<ImplicitCastExpr>()) {
        Current = ICE;
        continue;
      }

      // Member call on the vector: check it's a read-only method.
      if (const auto *MCE = Parent.get<CXXMemberCallExpr>()) {
        const CXXMethodDecl *Method = MCE->getMethodDecl();
        if (!Method || !isReadOnlyVectorMethod(Method->getName()))
          return false;
        break;
      }

      // Operator[] via CXXOperatorCallExpr.
      if (const auto *OCE = Parent.get<CXXOperatorCallExpr>()) {
        if (OCE->getOperator() == OO_Subscript)
          break;
        return false;
      }

      // Used in a range-based for loop: the DRE is inside the implicit
      // __range variable's initializer, so the parent is a VarDecl.
      if (const auto *VD = Parent.get<VarDecl>()) {
        if (VD->isImplicit()) {
          // Check that the implicit VarDecl is the range variable of a
          // CXXForRangeStmt.
          const auto VDParents = Context.getParents(*VD);
          for (const auto &VDP : VDParents) {
            if (const auto *DS = VDP.get<DeclStmt>()) {
              const auto DSParents = Context.getParents(*DS);
              for (const auto &DSP : DSParents)
                if (DSP.get<CXXForRangeStmt>())
                  goto range_ok;
            }
          }
        }
        return false;
      range_ok:
        break;
      }

      // Member expression (e.g. v.size()) - walk further up.
      if (Parent.get<MemberExpr>()) {
        Current = Parent.get<MemberExpr>();
        continue;
      }

      // Passed as argument to a function - check parameter type.
      if (const auto *CE = Parent.get<CallExpr>()) {
        const FunctionDecl *Callee = CE->getDirectCallee();
        if (!Callee)
          return false;
        // Find which argument position this is.
        bool Found = false;
        for (unsigned I = 0; I < CE->getNumArgs(); ++I) {
          if (CE->getArg(I)->IgnoreParenImpCasts() == DRE) {
            if (I < Callee->getNumParams()) {
              const QualType PT = Callee->getParamDecl(I)->getType();
              // Accept const vector<T>&, const T*, span<const T>.
              if (PT->isReferenceType() &&
                  PT.getNonReferenceType().isConstQualified()) {
                Found = true;
                break;
              }
              if (PT->isPointerType() &&
                  PT->getPointeeType().isConstQualified()) {
                Found = true;
                break;
              }
            }
            break;
          }
        }
        if (!Found)
          return false;
        break;
      }

      // Anything else is not read-only.
      return false;
    }
  }
  return true;
}

void UseSpanParamCheck::registerMatchers(MatchFinder *Finder) {
  // Match functions with const std::vector<T>& parameters.
  Finder->addMatcher(
      functionDecl(
          isDefinition(), unless(isExpansionInSystemHeader()),
          unless(isImplicit()), unless(isDeleted()),
          has(typeLoc(forEach(
              parmVarDecl(hasType(qualType(references(qualType(
                              isConstQualified(),
                              hasDeclaration(classTemplateSpecializationDecl(
                                  hasName("::std::vector"))))))))
                  .bind("param")))))
          .bind("func"),
      this);
}

void UseSpanParamCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  if (!Func || !Param)
    return;

  // Skip if this is a virtual function (can't change signature).
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Func))
    if (Method->isVirtual())
      return;

  // Skip if function has other overloads (changing signature is risky).
  // Skip template functions for now (type deduction complexity).
  if (Func->isTemplated())
    return;

  if (!allUsesAreReadOnly(Param, Func, *Result.Context))
    return;

  // Determine the element type from vector<T>.
  const QualType ParamType = Param->getType().getNonReferenceType();
  const auto *Spec =
      dyn_cast<ClassTemplateSpecializationDecl>(ParamType->getAsRecordDecl());
  if (!Spec || Spec->getTemplateArgs().size() < 1)
    return;

  const QualType ElemType = Spec->getTemplateArgs()[0].getAsType();
  const std::string SpanType =
      "std::span<const " + ElemType.getAsString() + ">";

  diag(Param->getLocation(),
       "parameter %0 can be changed to 'std::span'; it is only used for "
       "read-only access")
      << Param
      << FixItHint::CreateReplacement(
             Param->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
             SpanType);
}

} // namespace clang::tidy::modernize
