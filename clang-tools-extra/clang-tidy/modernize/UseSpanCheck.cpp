//===--- UseSpanCheck.cpp - clang-tidy ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSpanCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "../utils/IncludeInserter.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
AST_MATCHER(QualType, isRefToVectorOrArray) {
  if (!Node->isReferenceType())
    return false;

  QualType PointeeType = Node->getPointeeType();

  const Type *UnqualifiedType = PointeeType.getTypePtr()->getUnqualifiedDesugaredType();
  if (!UnqualifiedType || !UnqualifiedType->isRecordType())
    return false;

  const CXXRecordDecl *Record = UnqualifiedType->getAsCXXRecordDecl();
  if (!Record)
    return false;

  const std::string Name = Record->getQualifiedNameAsString();
  return Name == "std::vector" || Name == "std::array";
}
} // namespace

UseSpanCheck::UseSpanCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(utils::IncludeSorter::IS_LLVM, areDiagsSelfContained()) {}

void UseSpanCheck::registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                                       Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseSpanCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(
          forEachDescendant(
              parmVarDecl(hasType(qualType(isRefToVectorOrArray())))
                  .bind("param"))),
      this);
}

void UseSpanCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  if (!Param)
    return;

  QualType ParamType = Param->getType();
  if (!ParamType->isReferenceType())
    return;

  // Get the pointee type (the vector/array type)
  QualType PointeeType = ParamType->getPointeeType();
  bool IsConst = PointeeType.isConstQualified();

  const Type *UnqualifiedType = PointeeType.getTypePtr()->getUnqualifiedDesugaredType();
  if (!UnqualifiedType || !UnqualifiedType->isRecordType())
    return;

  const CXXRecordDecl *Record = UnqualifiedType->getAsCXXRecordDecl();
  if (!Record)
    return;

  const std::string RecordName = Record->getQualifiedNameAsString();
  if (RecordName != "std::vector" && RecordName != "std::array")
    return;

  // Check if it's a template specialization
  if (!isa<ClassTemplateSpecializationDecl>(Record))
    return;

  // Get the template arguments
  const auto *TemplateSpecRecord = cast<ClassTemplateSpecializationDecl>(Record);
  const TemplateArgumentList &Args = TemplateSpecRecord->getTemplateArgs();
  if (Args.size() < 1)
    return;

  // Get the element type from the first template argument
  const TemplateArgument &Arg = Args[0];
  if (Arg.getKind() != TemplateArgument::Type)
    return;

  QualType ElementType = Arg.getAsType();

  // Get the source range for the parameter type
  TypeSourceInfo *TSI = Param->getTypeSourceInfo();
  TypeLoc TL = TSI->getTypeLoc();

  // Get the source range for the entire type, including qualifiers
  SourceRange TypeRange = TL.getSourceRange();

  // Create the diagnostic
  auto Diag = diag(Param->getBeginLoc(),
                  "parameter %0 is reference to %1; consider using std::span instead")
      << Param
      << (RecordName == "std::vector" ? "std::vector" : "std::array");

  // Create the fix-it hint
  std::string SpanType;
  std::string ElementTypeWithConst = IsConst ? "const " + ElementType.getAsString() : ElementType.getAsString();

  // For std::array, we should preserve the size in the span
  if (RecordName == "std::array" && Args.size() >= 2) {
    const TemplateArgument &SizeArg = Args[1];
    if (SizeArg.getKind() == TemplateArgument::Integral) {
      llvm::APSInt Size = SizeArg.getAsIntegral();
      // Convert APSInt to string
      SmallString<16> SizeStr;
      Size.toString(SizeStr, 10);
      SpanType = "std::span<" + ElementTypeWithConst + ", " + SizeStr.str().str() + ">";
    } else {
      SpanType = "std::span<" + ElementTypeWithConst + ">";
    }
  } else {
    SpanType = "std::span<" + ElementTypeWithConst + ">";
  }

  // Create the replacement for the entire type including qualifiers
  Diag << FixItHint::CreateReplacement(TypeRange, SpanType);

  // Add the include for <span>
  Diag << Inserter.createIncludeInsertion(
      Result.SourceManager->getFileID(Param->getBeginLoc()),
      "<span>");
}

} // namespace clang::tidy::modernize
