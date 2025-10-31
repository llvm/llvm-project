//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstCorrectnessCheck.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <cassert>

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
      AnalyzePointers(Options.get("AnalyzePointers", true)),
      AnalyzeReferences(Options.get("AnalyzeReferences", true)),
      AnalyzeValues(Options.get("AnalyzeValues", true)),

      WarnPointersAsPointers(Options.get("WarnPointersAsPointers", true)),
      WarnPointersAsValues(Options.get("WarnPointersAsValues", false)),

      TransformPointersAsPointers(
          Options.get("TransformPointersAsPointers", true)),
      TransformPointersAsValues(
          Options.get("TransformPointersAsValues", false)),
      TransformReferences(Options.get("TransformReferences", true)),
      TransformValues(Options.get("TransformValues", true)),

      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {
  if (AnalyzeValues == false && AnalyzeReferences == false &&
      AnalyzePointers == false)
    this->configurationDiag(
        "The check 'misc-const-correctness' will not "
        "perform any analysis because 'AnalyzeValues', "
        "'AnalyzeReferences' and 'AnalyzePointers' are false.");
}

void ConstCorrectnessCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AnalyzePointers", AnalyzePointers);
  Options.store(Opts, "AnalyzeReferences", AnalyzeReferences);
  Options.store(Opts, "AnalyzeValues", AnalyzeValues);

  Options.store(Opts, "WarnPointersAsPointers", WarnPointersAsPointers);
  Options.store(Opts, "WarnPointersAsValues", WarnPointersAsValues);

  Options.store(Opts, "TransformPointersAsPointers",
                TransformPointersAsPointers);
  Options.store(Opts, "TransformPointersAsValues", TransformPointersAsValues);
  Options.store(Opts, "TransformReferences", TransformReferences);
  Options.store(Opts, "TransformValues", TransformValues);

  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

void ConstCorrectnessCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConstType =
      hasType(qualType(isConstQualified(),
                       // pointee check will check the constness of pointer
                       unless(pointerType())));

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

  auto AllowedTypeDecl = namedDecl(
      anyOf(matchers::matchesAnyListedName(AllowedTypes), usingShadowDecl()));

  const auto AllowedType = hasType(qualType(
      anyOf(hasDeclaration(AllowedTypeDecl), references(AllowedTypeDecl),
            pointerType(pointee(hasDeclaration(AllowedTypeDecl))))));

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
                   hasType(cxxRecordDecl(isLambda())), isImplicit(),
                   AllowedType)));

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
  const auto *VarDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("decl-stmt");
  // It can not be guaranteed that the variable is declared isolated,
  // therefore a transformation might effect the other variables as well and
  // be incorrect.
  const bool CanBeFixIt = VarDeclStmt != nullptr && VarDeclStmt->isSingleDecl();

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
  const QualType VT = Variable->getType();
  if (VT->isReferenceType()) {
    VC = VariableCategory::Reference;
  } else if (VT->isPointerType()) {
    VC = VariableCategory::Pointer;
  } else if (const auto *ArrayT = dyn_cast<ArrayType>(VT)) {
    if (ArrayT->getElementType()->isPointerType())
      VC = VariableCategory::Pointer;
  }

  auto CheckValue = [&]() {
    // The scope is only registered if the analysis shall be run.
    registerScope(LocalScope, Result.Context);

    // Offload const-analysis to utility function.
    if (ScopesCache[LocalScope]->isMutated(Variable))
      return;

    auto Diag = diag(Variable->getBeginLoc(),
                     "variable %0 of type %1 can be declared 'const'")
                << Variable << VT;
    if (IsNormalVariableInTemplate)
      TemplateDiagnosticsCache.insert(Variable->getBeginLoc());
    if (!CanBeFixIt)
      return;
    using namespace utils::fixit;
    if (VC == VariableCategory::Value && TransformValues) {
      Diag << addQualifierToVarDecl(*Variable, *Result.Context,
                                    Qualifiers::Const, QualifierTarget::Value,
                                    QualifierPolicy::Right);
      // FIXME: Add '{}' for default initialization if no user-defined default
      // constructor exists and there is no initializer.
      return;
    }

    if (VC == VariableCategory::Reference && TransformReferences) {
      Diag << addQualifierToVarDecl(*Variable, *Result.Context,
                                    Qualifiers::Const, QualifierTarget::Value,
                                    QualifierPolicy::Right);
      return;
    }

    if (VC == VariableCategory::Pointer && TransformPointersAsValues) {
      Diag << addQualifierToVarDecl(*Variable, *Result.Context,
                                    Qualifiers::Const, QualifierTarget::Value,
                                    QualifierPolicy::Right);
      return;
    }
  };

  auto CheckPointee = [&]() {
    assert(VC == VariableCategory::Pointer);
    registerScope(LocalScope, Result.Context);
    if (ScopesCache[LocalScope]->isPointeeMutated(Variable))
      return;
    auto Diag =
        diag(Variable->getBeginLoc(),
             "pointee of variable %0 of type %1 can be declared 'const'")
        << Variable << VT;
    if (IsNormalVariableInTemplate)
      TemplateDiagnosticsCache.insert(Variable->getBeginLoc());
    if (!CanBeFixIt)
      return;
    using namespace utils::fixit;
    if (TransformPointersAsPointers) {
      Diag << addQualifierToVarDecl(*Variable, *Result.Context,
                                    Qualifiers::Const, QualifierTarget::Pointee,
                                    QualifierPolicy::Right);
    }
  };

  // Each variable can only be in one category: Value, Pointer, Reference.
  // Analysis can be controlled for every category.
  if (VC == VariableCategory::Value && AnalyzeValues) {
    CheckValue();
    return;
  }
  if (VC == VariableCategory::Reference && AnalyzeReferences) {
    if (VT->getPointeeType()->isPointerType() && !WarnPointersAsValues)
      return;
    CheckValue();
    return;
  }
  if (VC == VariableCategory::Pointer && AnalyzePointers) {
    if (WarnPointersAsValues && !VT.isConstQualified())
      CheckValue();
    if (WarnPointersAsPointers) {
      if (const auto *PT = dyn_cast<PointerType>(VT)) {
        if (!PT->getPointeeType().isConstQualified() &&
            !PT->getPointeeType()->isFunctionType())
          CheckPointee();
      }
      if (const auto *AT = dyn_cast<ArrayType>(VT)) {
        if (!AT->getElementType().isConstQualified()) {
          assert(AT->getElementType()->isPointerType());
          CheckPointee();
        }
      }
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
