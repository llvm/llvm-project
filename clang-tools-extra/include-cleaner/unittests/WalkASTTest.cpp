#include "AnalysisInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Testing/TestAST.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"

namespace clang {
namespace include_cleaner {
namespace {

// Specifies a test of which symbols are referenced by a piece of code.
//
// Example:
//   Target:      int ^foo();
//   Referencing: int x = ^foo();
// There must be exactly one referencing location marked.
void testWalk(llvm::StringRef TargetCode, llvm::StringRef ReferencingCode) {
  llvm::Annotations Target(TargetCode);
  llvm::Annotations Referencing(ReferencingCode);

  TestInputs Inputs(Referencing.code());
  Inputs.ExtraFiles["target.h"] = Target.code().str();
  Inputs.ExtraArgs.push_back("-include");
  Inputs.ExtraArgs.push_back("target.h");
  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();

  // We're only going to record references from the nominated point,
  // to the target file.
  FileID ReferencingFile = SM.getMainFileID();
  SourceLocation ReferencingLoc =
      SM.getComposedLoc(ReferencingFile, Referencing.point());
  FileID TargetFile = SM.translateFile(
      llvm::cantFail(AST.fileManager().getFileRef("target.h")));

  // Perform the walk, and capture the offsets of the referenced targets.
  std::vector<size_t> ReferencedOffsets;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    if (ReferencingFile != SM.getDecomposedExpansionLoc(D->getLocation()).first)
      continue;
    walkAST(*D, [&](SourceLocation Loc, NamedDecl &ND) {
      if (SM.getFileLoc(Loc) != ReferencingLoc)
        return;
      auto NDLoc = SM.getDecomposedLoc(SM.getFileLoc(ND.getLocation()));
      if (NDLoc.first != TargetFile)
        return;
      ReferencedOffsets.push_back(NDLoc.second);
    });
  }
  llvm::sort(ReferencedOffsets);

  // Compare results to the expected points.
  // For each difference, show the target point in context, like a diagnostic.
  std::string DiagBuf;
  llvm::raw_string_ostream DiagOS(DiagBuf);
  auto *DiagOpts = new DiagnosticOptions();
  DiagOpts->ShowLevel = 0;
  DiagOpts->ShowNoteIncludeStack = 0;
  TextDiagnostic Diag(DiagOS, AST.context().getLangOpts(), DiagOpts);
  auto DiagnosePoint = [&](const char *Message, unsigned Offset) {
    Diag.emitDiagnostic(
        FullSourceLoc(SM.getComposedLoc(TargetFile, Offset), SM),
        DiagnosticsEngine::Note, Message, {}, {});
  };
  for (auto Expected : Target.points())
    if (!llvm::is_contained(ReferencedOffsets, Expected))
      DiagnosePoint("location not marked used", Expected);
  for (auto Actual : ReferencedOffsets)
    if (!llvm::is_contained(Target.points(), Actual))
      DiagnosePoint("location unexpectedly used", Actual);

  // If there were any differences, we print the entire referencing code once.
  if (!DiagBuf.empty())
    ADD_FAILURE() << DiagBuf << "\nfrom code:\n" << ReferencingCode;
}

TEST(WalkAST, DeclRef) {
  testWalk("int ^x;", "int y = ^x;");
  testWalk("int ^foo();", "int y = ^foo();");
  testWalk("namespace ns { int ^x; }", "int y = ns::^x;");
  testWalk("struct S { static int ^x; };", "int y = S::^x;");
  // Canonical declaration only.
  testWalk("extern int ^x; int x;", "int y = ^x;");
}

TEST(WalkAST, TagType) {
  testWalk("struct ^S {};", "^S *y;");
  testWalk("enum ^E {};", "^E *y;");
  testWalk("struct ^S { static int x; };", "int y = ^S::x;");
}

TEST(WalkAST, Alias) {
  testWalk(R"cpp(
    namespace ns { int x; }
    using ns::^x;
  )cpp",
           "int y = ^x;");
  testWalk(R"cpp(
    namespace ns { struct S; } // Not used
    using ns::S; // FIXME: S should be used
  )cpp",
           "^S *x;");
}

} // namespace
} // namespace include_cleaner
} // namespace clang
