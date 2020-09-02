//===--- ConstCorrectnessCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../utils/FixItHintUtils.h"
#include "ConstCorrectnessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

namespace {
// FIXME: This matcher exists in some other code-review as well.
// It should probably move to ASTMatchers.
AST_MATCHER(VarDecl, isLocal) { return Node.isLocalVarDecl(); }
AST_MATCHER_P(DeclStmt, containsDeclaration2,
              ast_matchers::internal::Matcher<Decl>, InnerMatcher) {
  return ast_matchers::internal::matchesFirstInPointerRange(
      InnerMatcher, Node.decl_begin(), Node.decl_end(), Finder, Builder);
}
AST_MATCHER(ReferenceType, isSpelledAsLValue) {
  return Node.isSpelledAsLValue();
}
AST_MATCHER(Type, isDependentType) { return Node.isDependentType(); }
} // namespace

void ConstCorrectnessCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AnalyzeValues", AnalyzeValues);
  Options.store(Opts, "AnalyzeReferences", AnalyzeReferences);
  Options.store(Opts, "WarnPointersAsValues", WarnPointersAsValues);

  Options.store(Opts, "TransformValues", TransformValues);
  Options.store(Opts, "TransformReferences", TransformReferences);
  Options.store(Opts, "TransformPointersAsValues", TransformPointersAsValues);
}

void ConstCorrectnessCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConstType = hasType(isConstQualified());
  const auto ConstReference = hasType(references(isConstQualified()));
  const auto RValueReference = hasType(
      referenceType(anyOf(rValueReferenceType(), unless(isSpelledAsLValue()))));

  const auto TemplateType = anyOf(
      hasType(hasCanonicalType(templateTypeParmType())),
      hasType(substTemplateTypeParmType()), hasType(isDependentType()),
      // References to template types, their substitutions or typedefs to
      // template types need to be considered as well.
      hasType(referenceType(pointee(hasCanonicalType(templateTypeParmType())))),
      hasType(referenceType(pointee(substTemplateTypeParmType()))));

  const auto AutoTemplateType = varDecl(
      anyOf(hasType(autoType()), hasType(referenceType(pointee(autoType()))),
            hasType(pointerType(pointee(autoType())))),
      hasInitializer(isInstantiationDependent()));

  const auto FunctionPointerRef =
      hasType(hasCanonicalType(referenceType(pointee(functionType()))));

  // Match local variables which could be 'const' if not modified later.
  // Example: `int i = 10` would match `int i`.
  const auto LocalValDecl = varDecl(
      allOf(isLocal(), hasInitializer(anything()),
            unless(anyOf(ConstType, ConstReference, TemplateType,
                         AutoTemplateType, RValueReference, FunctionPointerRef,
                         hasType(cxxRecordDecl(isLambda())), isImplicit()))));

  // Match the function scope for which the analysis of all local variables
  // shall be run.
  const auto FunctionScope = functionDecl(hasBody(
      compoundStmt(findAll(declStmt(allOf(containsDeclaration2(
                                              LocalValDecl.bind("local-value")),
                                          unless(has(decompositionDecl()))))
                               .bind("decl-stmt")))
          .bind("scope")));

  Finder->addMatcher(FunctionScope, this);
}

/// Classify for a variable in what the Const-Check is interested.
enum class VariableCategory { Value, Reference, Pointer };

void ConstCorrectnessCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *LocalScope = Result.Nodes.getNodeAs<CompoundStmt>("scope");
  assert(LocalScope && "Did not match scope for local variable");
  registerScope(LocalScope, Result.Context);

  const auto *Variable = Result.Nodes.getNodeAs<VarDecl>("local-value");
  assert(Variable && "Did not match local variable definition");

  VariableCategory VC = VariableCategory::Value;
  if (Variable->getType()->isReferenceType())
    VC = VariableCategory::Reference;
  if (Variable->getType()->isPointerType())
    VC = VariableCategory::Pointer;

  // Each variable can only in one category: Value, Pointer, Reference.
  // Analysis can be controlled for every category.
  if (VC == VariableCategory::Reference && !AnalyzeReferences)
    return;

  if (VC == VariableCategory::Reference &&
      Variable->getType()->getPointeeType()->isPointerType() &&
      !WarnPointersAsValues)
    return;

  if (VC == VariableCategory::Pointer && !WarnPointersAsValues)
    return;

  if (VC == VariableCategory::Value && !AnalyzeValues)
    return;

  // Offload const-analysis to utility function.
  if (ScopesCache[LocalScope]->isMutated(Variable))
    return;

  auto Diag = diag(Variable->getBeginLoc(),
                   "variable %0 of type %1 can be declared 'const'")
              << Variable << Variable->getType();

  const auto *VarDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("decl-stmt");

  // It can not be guaranteed that the variable is declared isolated, therefore
  // a transformation might effect the other variables as well and be incorrect.
  if (VarDeclStmt == nullptr || !VarDeclStmt->isSingleDecl())
    return;

  using namespace utils::fixit;
  using llvm::Optional;
  if (VC == VariableCategory::Value && TransformValues) {
    if (Optional<FixItHint> Fix = addQualifierToVarDecl(
            *Variable, *Result.Context, DeclSpec::TQ_const,
            QualifierTarget::Value, QualifierPolicy::Left)) {
      Diag << *Fix;
      // FIXME: Add '{}' for default initialization if no user-defined default
      // constructor exists and there is no initializer.
    }
    return;
  }

  if (VC == VariableCategory::Reference && TransformReferences) {
    if (Optional<FixItHint> Fix = addQualifierToVarDecl(
            *Variable, *Result.Context, DeclSpec::TQ_const,
            QualifierTarget::Value, QualifierPolicy::Left))
      Diag << *Fix;
    return;
  }

  if (VC == VariableCategory::Pointer) {
    if (WarnPointersAsValues && TransformPointersAsValues) {
      if (Optional<FixItHint> Fix = addQualifierToVarDecl(
              *Variable, *Result.Context, DeclSpec::TQ_const,
              QualifierTarget::Value, QualifierPolicy::Left))
        Diag << *Fix;
    }
    return;
  }
}

void ConstCorrectnessCheck::registerScope(const CompoundStmt *LocalScope,
                                          ASTContext *Context) {
  if (ScopesCache.find(LocalScope) == ScopesCache.end())
    ScopesCache.insert(std::make_pair(
        LocalScope,
        std::make_unique<ExprMutationAnalyzer>(*LocalScope, *Context)));
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
