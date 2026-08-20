//===- TestTransformation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation used only by lit tests for the source-edit-generation
// pipeline. It walks every function in the main source file, emits a
// zero-length `/*T*/` comment at the function body's start, and adds one
// `test-touches-function` note per visited function. Its level is always
// `Note` — the goal is to exercise the framework's plumbing, not to
// produce meaningful findings. The level rises to `Warning` when the
// input WPASuite's id table is non-empty, giving lit tests a knob to
// confirm the suite is read at all without depending on namespace
// matching.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformation.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace clang::ssaf;

namespace {

class TestTransformation final : public Transformation {
public:
  using Transformation::Transformation;

  void HandleTranslationUnit(ASTContext &Ctx) override {
    bool SuiteIsNonEmpty = Suite.getIdTable().count() > 0;
    Visitor V{*this, Ctx, SuiteIsNonEmpty};
    V.TraverseDecl(Ctx.getTranslationUnitDecl());
  }

private:
  class Visitor : public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(TestTransformation &T, ASTContext &Ctx, bool SuiteIsNonEmpty)
        : T(T), Ctx(Ctx), Level(SuiteIsNonEmpty
                                    ? clang::SarifResultLevel::Warning
                                    : clang::SarifResultLevel::Note) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
      if (!FD->hasBody())
        return true;
      SourceManager &SM = Ctx.getSourceManager();
      if (!SM.isInMainFile(FD->getLocation()))
        return true;

      Stmt *Body = FD->getBody();
      SourceLocation BodyStart = Body->getBeginLoc();
      if (BodyStart.isInvalid())
        return true;

      llvm::SmallString<64> FilePath(SM.getFilename(BodyStart));
      unsigned Offset = SM.getFileOffset(BodyStart);
      T.Edits.addReplacement(
          clang::tooling::Replacement(FilePath, Offset, /*Length=*/0, "/*T*/"));

      CharSourceRange Range = Lexer::getAsCharRange(
          CharSourceRange::getTokenRange(FD->getNameInfo().getSourceRange()),
          SM, Ctx.getLangOpts());
      std::string Message = "visited " + FD->getNameAsString();
      T.Report.addResult("test-touches-function", Level, Range, Message);
      return true;
    }

  private:
    TestTransformation &T;
    ASTContext &Ctx;
    clang::SarifResultLevel Level;
  };
};

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SSAFTestTransformationAnchorSource = 0;
} // namespace clang::ssaf

// This global causes issue in stage2 with ASan-instrumented clang so
// adding the no-ASan attribute.
static TransformationRegistry::Add<TestTransformation>
    __attribute__((no_sanitize("address")))
    RegisterTestTransformation("test-transformation",
                               "Test transformation for the SSAF "
                               "source-edit-generation lit suite");

