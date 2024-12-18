//===--- ConstCorrectnessCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstCorrectnessCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {
// FIXME: This matcher exists in some other code-review as well.
// It should probably move to ASTMatchers.
AST_MATCHER(VarDecl, isLocal) { return Node.isLocalVarDecl(); }
AST_MATCHER_P(DeclStmt, containsAnyDeclaration,
              ast_matchers::internal::Matcher<Decl>, InnerMatcher) {
  return ast_matchers::internal::matchesFirstInPointerRange(
             InnerMatcher, Node.decl_begin(), Node.decl_end(), Finder,
             Builder) != Node.decl_end();
}
AST_MATCHER(ReferenceType, isSpelledAsLValue) {
  return Node.isSpelledAsLValue();
}
AST_MATCHER(Type, isDependentType) { return Node.isDependentType(); }
} // namespace

ConstCorrectnessCheck::ConstCorrectnessCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AnalyzeValues(Options.get("AnalyzeValues", true)),
      AnalyzeReferences(Options.get("AnalyzeReferences", true)),
      WarnPointersAsValues(Options.get("WarnPointersAsValues", false)),
      TransformValues(Options.get("TransformValues", true)),
      TransformReferences(Options.get("TransformReferences", true)),
      TransformPointersAsValues(
          Options.get("TransformPointersAsValues", false)) {
  if (AnalyzeValues == false && AnalyzeReferences == false)
    this->configurationDiag(
        "The check 'misc-const-correctness' will not "
        "perform any analysis because both 'AnalyzeValues' and "
        "'AnalyzeReferences' are false.");
}

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
            hasType(pointerType(pointee(autoType())))));

  const auto FunctionPointerRef =
      hasType(hasCanonicalType(referenceType(pointee(functionType()))));

  // Match local variables which could be 'const' if not modified later.
  // Example: `int i = 10` would match `int i`.
  const auto LocalValDecl = varDecl(
      isLocal(), hasInitializer(anything()),
      unless(anyOf(ConstType, ConstReference, TemplateType,
                   hasInitializer(isInstantiationDependent()), AutoTemplateType,
                   RValueReference, FunctionPointerRef,
                   hasType(cxxRecordDecl(isLambda())), isImplicit())));

  // Match the function scope for which the analysis of all local variables
  // shall be run.
  const auto FunctionScope =
      functionDecl(
          hasBody(stmt(forEachDescendant(
                           declStmt(containsAnyDeclaration(
                                        LocalValDecl.bind("local-value")),
                                    unless(has(decompositionDecl())))
                               .bind("decl-stmt")))
                      .bind("scope")))
          .bind("function-decl");

  Finder->addMatcher(FunctionScope, this);
}

/// Classify for a variable in what the Const-Check is interested.
enum class VariableCategory { Value, Reference, Pointer };

void ConstCorrectnessCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *LocalScope = Result.Nodes.getNodeAs<Stmt>("scope");
  const auto *Variable = Result.Nodes.getNodeAs<VarDecl>("local-value");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function-decl");

  /// If the variable was declared in a template it might be analyzed multiple
  /// times. Only one of those instantiations shall emit a warning. NOTE: This
  /// shall only deduplicate warnings for variables that are not instantiation
  /// dependent. Variables like 'int x = 42;' in a template that can become
  /// const emit multiple warnings otherwise.
  bool IsNormalVariableInTemplate = Function->isTemplateInstantiation();
  if (IsNormalVariableInTemplate &&
      TemplateDiagnosticsCache.contains(Variable->getBeginLoc()))
    return;

  VariableCategory VC = VariableCategory::Value;
  if (Variable->getType()->isReferenceType())
    VC = VariableCategory::Reference;
  if (Variable->getType()->isPointerType())
    VC = VariableCategory::Pointer;
  if (Variable->getType()->isArrayType()) {
    if (const auto *ArrayT = dyn_cast<ArrayType>(Variable->getType())) {
      if (ArrayT->getElementType()->isPointerType())
        VC = VariableCategory::Pointer;
    }
  }

  // Each variable can only be in one category: Value, Pointer, Reference.
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

  // The scope is only registered if the analysis shall be run.
  registerScope(LocalScope, Result.Context);

  // Offload const-analysis to utility function.
  if (ScopesCache[LocalScope]->isMutated(Variable))
    return;

  auto Diag = diag(Variable->getBeginLoc(),
                   "variable %0 of type %1 can be declared 'const'")
              << Variable << Variable->getType();
  if (IsNormalVariableInTemplate)
    TemplateDiagnosticsCache.insert(Variable->getBeginLoc());

  const auto *VarDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("decl-stmt");

  // It can not be guaranteed that the variable is declared isolated, therefore
  // a transformation might effect the other variables as well and be incorrect.
  if (VarDeclStmt == nullptr || !VarDeclStmt->isSingleDecl())
    return;

  using namespace utils::fixit;
  if (VC == VariableCategory::Value && TransformValues) {
    Diag << addQualifierToVarDecl(*Variable, *Result.Context, Qualifiers::Const,
                                  QualifierTarget::Value,
                                  QualifierPolicy::Right);
    // FIXME: Add '{}' for default initialization if no user-defined default
    // constructor exists and there is no initializer.
    return;
  }

  if (VC == VariableCategory::Reference && TransformReferences) {
    Diag << addQualifierToVarDecl(*Variable, *Result.Context, Qualifiers::Const,
                                  QualifierTarget::Value,
                                  QualifierPolicy::Right);
    return;
  }

  if (VC == VariableCategory::Pointer) {
    if (WarnPointersAsValues && TransformPointersAsValues) {
      Diag << addQualifierToVarDecl(*Variable, *Result.Context,
                                    Qualifiers::Const, QualifierTarget::Value,
                                    QualifierPolicy::Right);
    }
    return;
  }
}

void ConstCorrectnessCheck::registerScope(const Stmt *LocalScope,
                                          ASTContext *Context) {
  auto &Analyzer = ScopesCache[LocalScope];
  if (!Analyzer)
    Analyzer = std::make_unique<ExprMutationAnalyzer>(*LocalScope, *Context);
}

} // namespace clang::tidy::misc
