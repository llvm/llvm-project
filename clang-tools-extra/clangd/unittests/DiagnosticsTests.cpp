//===--- DiagnosticsTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdUnit.h"
#include "Diagnostics.h"
#include "Path.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/MemIndex.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

using testing::_;
using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

testing::Matcher<const Diag &> WithFix(testing::Matcher<Fix> FixMatcher) {
  return Field(&Diag::Fixes, ElementsAre(FixMatcher));
}

testing::Matcher<const Diag &> WithFix(testing::Matcher<Fix> FixMatcher1,
                                       testing::Matcher<Fix> FixMatcher2) {
  return Field(&Diag::Fixes, UnorderedElementsAre(FixMatcher1, FixMatcher2));
}

testing::Matcher<const Diag &> WithNote(testing::Matcher<Note> NoteMatcher) {
  return Field(&Diag::Notes, ElementsAre(NoteMatcher));
}

MATCHER_P2(Diag, Range, Message,
           "Diag at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Range == Range && arg.Message == Message;
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Message == Message && arg.Edits.size() == 1 &&
         arg.Edits[0].range == Range && arg.Edits[0].newText == Replacement;
}

MATCHER_P(EqualToLSPDiag, LSPDiag,
          "LSP diagnostic " + llvm::to_string(LSPDiag)) {
  if (toJSON(arg) != toJSON(LSPDiag)) {
    *result_listener << llvm::formatv("expected:\n{0:2}\ngot\n{1:2}",
                                      toJSON(LSPDiag), toJSON(arg))
                            .str();
    return false;
  }
  return true;
}

MATCHER_P(DiagSource, S, "") { return arg.Source == S; }
MATCHER_P(DiagName, N, "") { return arg.Name == N; }

MATCHER_P(EqualToFix, Fix, "LSP fix " + llvm::to_string(Fix)) {
  if (arg.Message != Fix.Message)
    return false;
  if (arg.Edits.size() != Fix.Edits.size())
    return false;
  for (std::size_t I = 0; I < arg.Edits.size(); ++I) {
    if (arg.Edits[I].range != Fix.Edits[I].range ||
        arg.Edits[I].newText != Fix.Edits[I].newText)
      return false;
  }
  return true;
}

// Helper function to make tests shorter.
Position pos(int line, int character) {
  Position Res;
  Res.line = line;
  Res.character = character;
  return Res;
}

TEST(DiagnosticsTest, DiagnosticRanges) {
  // Check we report correct ranges, including various edge-cases.
  Annotations Test(R"cpp(
    namespace test{};
    void $decl[[foo]]();
    int main() {
      $typo[[go\
o]]();
      foo()$semicolon[[]]//with comments
      $unk[[unknown]]();
      double $type[[bar]] = "foo";
      struct Foo { int x; }; Foo a;
      a.$nomember[[y]];
      test::$nomembernamespace[[test]];
    }
  )cpp");
  EXPECT_THAT(
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(
          // This range spans lines.
          AllOf(Diag(Test.range("typo"),
                     "use of undeclared identifier 'goo'; did you mean 'foo'?"),
                DiagSource(Diag::Clang), DiagName("undeclared_var_use_suggest"),
                WithFix(
                    Fix(Test.range("typo"), "foo", "change 'go\\ o' to 'foo'")),
                // This is a pretty normal range.
                WithNote(Diag(Test.range("decl"), "'foo' declared here"))),
          // This range is zero-width and insertion. Therefore make sure we are
          // not expanding it into other tokens. Since we are not going to
          // replace those.
          AllOf(Diag(Test.range("semicolon"), "expected ';' after expression"),
                WithFix(Fix(Test.range("semicolon"), ";", "insert ';'"))),
          // This range isn't provided by clang, we expand to the token.
          Diag(Test.range("unk"), "use of undeclared identifier 'unknown'"),
          Diag(Test.range("type"),
               "cannot initialize a variable of type 'double' with an lvalue "
               "of type 'const char [4]'"),
          Diag(Test.range("nomember"), "no member named 'y' in 'Foo'"),
          Diag(Test.range("nomembernamespace"),
               "no member named 'test' in namespace 'test'")));
}

TEST(DiagnosticsTest, FlagsMatter) {
  Annotations Test("[[void]] main() {}");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                WithFix(Fix(Test.range(), "int",
                                            "change 'void' to 'int'")))));
  // Same code built as C gets different diagnostics.
  TU.Filename = "Plain.c";
  EXPECT_THAT(
      TU.build().getDiagnostics(),
      ElementsAre(AllOf(
          Diag(Test.range(), "return type of 'main' is not 'int'"),
          WithFix(Fix(Test.range(), "int", "change return type to 'int'")))));
}

TEST(DiagnosticsTest, DiagnosticPreamble) {
  Annotations Test(R"cpp(
    #include $[["not-found.h"]]
  )cpp");

  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(TU.build().getDiagnostics(),
              ElementsAre(testing::AllOf(
                  Diag(Test.range(), "'not-found.h' file not found"),
                  DiagSource(Diag::Clang), DiagName("pp_file_not_found"))));
}

TEST(DiagnosticsTest, ClangTidy) {
  Annotations Test(R"cpp(
    #include $deprecated[["assert.h"]]

    #define $macrodef[[SQUARE]](X) (X)*(X)
    int main() {
      return $doubled[[sizeof]](sizeof(int));
      int y = 4;
      return SQUARE($macroarg[[++]]y);
    }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.HeaderFilename = "assert.h"; // Suppress "not found" error.
  TU.ClangTidyChecks =
      "-*, bugprone-sizeof-expression, bugprone-macro-repeated-side-effects, "
      "modernize-deprecated-headers";
  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("deprecated"),
                     "inclusion of deprecated C++ header 'assert.h'; consider "
                     "using 'cassert' instead"),
                DiagSource(Diag::ClangTidy),
                DiagName("modernize-deprecated-headers"),
                WithFix(Fix(Test.range("deprecated"), "<cassert>",
                            "change '\"assert.h\"' to '<cassert>'"))),
          Diag(Test.range("doubled"),
               "suspicious usage of 'sizeof(sizeof(...))'"),
          AllOf(
              Diag(Test.range("macroarg"),
                   "side effects in the 1st macro argument 'X' are repeated in "
                   "macro expansion"),
              DiagSource(Diag::ClangTidy),
              DiagName("bugprone-macro-repeated-side-effects"),
              WithNote(
                  Diag(Test.range("macrodef"), "macro 'SQUARE' defined here"))),
          Diag(Test.range("macroarg"),
               "multiple unsequenced modifications to 'y'")));
}

TEST(DiagnosticsTest, Preprocessor) {
  // This looks like a preamble, but there's an #else in the middle!
  // Check that:
  //  - the #else doesn't generate diagnostics (we had this bug)
  //  - we get diagnostics from the taken branch
  //  - we get no diagnostics from the not taken branch
  Annotations Test(R"cpp(
    #ifndef FOO
    #define FOO
      int a = [[b]];
    #else
      int x = y;
    #endif
    )cpp");
  EXPECT_THAT(
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'b'")));
}

TEST(DiagnosticsTest, InsideMacros) {
  Annotations Test(R"cpp(
    #define TEN 10
    #define RET(x) return x + 10

    int* foo() {
      RET($foo[[0]]);
    }
    int* bar() {
      return $bar[[TEN]];
    }
    )cpp");
  EXPECT_THAT(TestTU::withCode(Test.code()).build().getDiagnostics(),
              ElementsAre(Diag(Test.range("foo"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'"),
                          Diag(Test.range("bar"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'")));
}

TEST(DiagnosticsTest, NoFixItInMacro) {
  Annotations Test(R"cpp(
    #define Define(name) void name() {}

    [[Define]](main)
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                Not(WithFix(_)))));
}

TEST(DiagnosticsTest, ToLSP) {
  URIForFile MainFile =
      URIForFile::canonicalize(testPath("foo/bar/main.cpp"), "");
  URIForFile HeaderFile =
      URIForFile::canonicalize(testPath("foo/bar/header.h"), "");

  clangd::Diag D;
  D.ID = clang::diag::err_enum_class_reference;
  D.Name = "enum_class_reference";
  D.Source = clangd::Diag::Clang;
  D.Message = "something terrible happened";
  D.Range = {pos(1, 2), pos(3, 4)};
  D.InsideMainFile = true;
  D.Severity = DiagnosticsEngine::Error;
  D.File = "foo/bar/main.cpp";
  D.AbsFile = MainFile.file();

  clangd::Note NoteInMain;
  NoteInMain.Message = "declared somewhere in the main file";
  NoteInMain.Range = {pos(5, 6), pos(7, 8)};
  NoteInMain.Severity = DiagnosticsEngine::Remark;
  NoteInMain.File = "../foo/bar/main.cpp";
  NoteInMain.InsideMainFile = true;
  NoteInMain.AbsFile = MainFile.file();

  D.Notes.push_back(NoteInMain);

  clangd::Note NoteInHeader;
  NoteInHeader.Message = "declared somewhere in the header file";
  NoteInHeader.Range = {pos(9, 10), pos(11, 12)};
  NoteInHeader.Severity = DiagnosticsEngine::Note;
  NoteInHeader.File = "../foo/baz/header.h";
  NoteInHeader.InsideMainFile = false;
  NoteInHeader.AbsFile = HeaderFile.file();
  D.Notes.push_back(NoteInHeader);

  clangd::Fix F;
  F.Message = "do something";
  D.Fixes.push_back(F);

  // Diagnostics should turn into these:
  clangd::Diagnostic MainLSP;
  MainLSP.range = D.Range;
  MainLSP.severity = getSeverity(DiagnosticsEngine::Error);
  MainLSP.code = "enum_class_reference";
  MainLSP.source = "clang";
  MainLSP.message =
      R"(Something terrible happened (fix available)

main.cpp:6:7: remark: declared somewhere in the main file

../foo/baz/header.h:10:11:
note: declared somewhere in the header file)";

  clangd::Diagnostic NoteInMainLSP;
  NoteInMainLSP.range = NoteInMain.Range;
  NoteInMainLSP.severity = getSeverity(DiagnosticsEngine::Remark);
  NoteInMainLSP.message = R"(Declared somewhere in the main file

main.cpp:2:3: error: something terrible happened)";

  ClangdDiagnosticOptions Opts;
  // Transform diagnostics and check the results.
  std::vector<std::pair<clangd::Diagnostic, std::vector<clangd::Fix>>> LSPDiags;
  toLSPDiags(D, MainFile, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix> Fixes) {
               LSPDiags.push_back(
                   {std::move(LSPDiag),
                    std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
             });

  EXPECT_THAT(
      LSPDiags,
      ElementsAre(Pair(EqualToLSPDiag(MainLSP), ElementsAre(EqualToFix(F))),
                  Pair(EqualToLSPDiag(NoteInMainLSP), IsEmpty())));
  EXPECT_EQ(LSPDiags[0].first.code, "enum_class_reference");
  EXPECT_EQ(LSPDiags[0].first.source, "clang");
  EXPECT_EQ(LSPDiags[1].first.code, "");
  EXPECT_EQ(LSPDiags[1].first.source, "");

  // Same thing, but don't flatten notes into the main list.
  LSPDiags.clear();
  Opts.EmitRelatedLocations = true;
  toLSPDiags(D, MainFile, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix> Fixes) {
               LSPDiags.push_back(
                   {std::move(LSPDiag),
                    std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
             });
  MainLSP.message = "Something terrible happened (fix available)";
  DiagnosticRelatedInformation NoteInMainDRI;
  NoteInMainDRI.message = "Declared somewhere in the main file";
  NoteInMainDRI.location.range = NoteInMain.Range;
  NoteInMainDRI.location.uri = MainFile;
  MainLSP.relatedInformation = {NoteInMainDRI};
  DiagnosticRelatedInformation NoteInHeaderDRI;
  NoteInHeaderDRI.message = "Declared somewhere in the header file";
  NoteInHeaderDRI.location.range = NoteInHeader.Range;
  NoteInHeaderDRI.location.uri = HeaderFile;
  MainLSP.relatedInformation = {NoteInMainDRI, NoteInHeaderDRI};
  EXPECT_THAT(LSPDiags, ElementsAre(Pair(EqualToLSPDiag(MainLSP),
                                         ElementsAre(EqualToFix(F)))));
}

struct SymbolWithHeader {
  std::string QName;
  std::string DeclaringFile;
  std::string IncludeHeader;
};

std::unique_ptr<SymbolIndex>
buildIndexWithSymbol(llvm::ArrayRef<SymbolWithHeader> Syms) {
  SymbolSlab::Builder Slab;
  for (const auto &S : Syms) {
    Symbol Sym = cls(S.QName);
    Sym.Flags |= Symbol::IndexedForCodeCompletion;
    Sym.CanonicalDeclaration.FileURI = S.DeclaringFile.c_str();
    Sym.Definition.FileURI = S.DeclaringFile.c_str();
    Sym.IncludeHeaders.emplace_back(S.IncludeHeader, 1);
    Slab.insert(Sym);
  }
  return MemIndex::build(std::move(Slab).build(), RefSlab());
}

TEST(IncludeFixerTest, IncompleteType) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
  class X;
  $nested[[X::]]Nested n;
}
class Y : $base[[public ns::X]] {};
int main() {
  ns::X *x;
  x$access[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"ns::X", "unittest:///x.h", "\"x.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("nested"),
                     "incomplete type 'ns::X' named in nested name specifier"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("base"), "base class has incomplete type"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("access"),
                     "member access into incomplete type 'ns::X'"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X")))));
}

TEST(IncludeFixerTest, NoSuggestIncludeWhenNoDefinitionInHeader) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
  class X;
}
class Y : $base[[public ns::X]] {};
int main() {
  ns::X *x;
  x$access[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  Symbol Sym = cls("ns::X");
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  Sym.CanonicalDeclaration.FileURI = "unittest:///x.h";
  Sym.Definition.FileURI = "unittest:///x.cc";
  Sym.IncludeHeaders.emplace_back("\"x.h\"", 1);

  SymbolSlab::Builder Slab;
  Slab.insert(Sym);
  auto Index = MemIndex::build(std::move(Slab).build(), RefSlab());
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Test.range("base"), "base class has incomplete type"),
                  Diag(Test.range("access"),
                       "member access into incomplete type 'ns::X'")));
}

TEST(IncludeFixerTest, Typo) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
void foo() {
  $unqualified1[[X]] x;
  // No fix if the unresolved type is used as specifier. (ns::)X::Nested will be
  // considered the unresolved type.
  $unqualified2[[X]]::Nested n;
}
}
void bar() {
  ns::$qualified1[[X]] x; // ns:: is valid.
  ns::$qualified2[[X]](); // Error: no member in namespace

  ::$global[[Global]] glob;
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"ns::X", "unittest:///x.h", "\"x.h\""},
       SymbolWithHeader{"Global", "unittest:///global.h", "\"global.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("unqualified1"), "unknown type name 'X'"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          Diag(Test.range("unqualified2"), "use of undeclared identifier 'X'"),
          AllOf(Diag(Test.range("qualified1"),
                     "no type named 'X' in namespace 'ns'"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("qualified2"),
                     "no member named 'X' in namespace 'ns'"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("global"),
                     "no type named 'Global' in the global namespace"),
                WithFix(Fix(Test.range("insert"), "#include \"global.h\"\n",
                            "Add include \"global.h\" for symbol Global")))));
}

TEST(IncludeFixerTest, MultipleMatchedSymbols) {
  Annotations Test(R"cpp(
$insert[[]]namespace na {
namespace nb {
void foo() {
  $unqualified[[X]] x;
}
}
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"na::X", "unittest:///a.h", "\"a.h\""},
       SymbolWithHeader{"na::nb::X", "unittest:///b.h", "\"b.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Test.range("unqualified"), "unknown type name 'X'"),
                  WithFix(Fix(Test.range("insert"), "#include \"a.h\"\n",
                              "Add include \"a.h\" for symbol na::X"),
                          Fix(Test.range("insert"), "#include \"b.h\"\n",
                              "Add include \"b.h\" for symbol na::nb::X")))));
}

TEST(IncludeFixerTest, NoCrashMemebrAccess) {
  Annotations Test(R"cpp(
    struct X { int  xyz; };
    void g() { X x; x.$[[xy]] }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"na::X", "unittest:///a.h", "\"a.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(Diag(Test.range(), "no member named 'xy' in 'X'")));
}

TEST(IncludeFixerTest, UseCachedIndexResults) {
  // As index results for the identical request are cached, more than 5 fixes
  // are generated.
  Annotations Test(R"cpp(
$insert[[]]void foo() {
  $x1[[X]] x;
  $x2[[X]] x;
  $x3[[X]] x;
  $x4[[X]] x;
  $x5[[X]] x;
  $x6[[X]] x;
  $x7[[X]] x;
}

class X;
void bar(X *x) {
  x$a1[[->]]f();
  x$a2[[->]]f();
  x$a3[[->]]f();
  x$a4[[->]]f();
  x$a5[[->]]f();
  x$a6[[->]]f();
  x$a7[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index =
      buildIndexWithSymbol(SymbolWithHeader{"X", "unittest:///a.h", "\"a.h\""});
  TU.ExternalIndex = Index.get();

  auto Parsed = TU.build();
  for (const auto &D : Parsed.getDiagnostics()) {
    EXPECT_EQ(D.Fixes.size(), 1u);
    EXPECT_EQ(D.Fixes[0].Message,
              std::string("Add include \"a.h\" for symbol X"));
  }
}

TEST(IncludeFixerTest, UnresolvedNameAsSpecifier) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
}
void g() {  ns::$[[scope]]::X_Y();  }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"ns::scope::X_Y", "unittest:///x.h", "\"x.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(AllOf(
          Diag(Test.range(), "no member named 'scope' in namespace 'ns'"),
          WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                      "Add include \"x.h\" for symbol ns::scope::X_Y")))));
}

TEST(IncludeFixerTest, UnresolvedSpecifierWithSemaCorrection) {
  Annotations Test(R"cpp(
$insert[[]]namespace clang {
void f() {
  // "clangd::" will be corrected to "clang::" by Sema.
  $q1[[clangd]]::$x[[X]] x;
  $q2[[clangd]]::$ns[[ns]]::Y y;
}
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"clang::clangd::X", "unittest:///x.h", "\"x.h\""},
       SymbolWithHeader{"clang::clangd::ns::Y", "unittest:///y.h", "\"y.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(
              Diag(Test.range("q1"), "use of undeclared identifier 'clangd'; "
                                     "did you mean 'clang'?"),
              WithFix(_, // change clangd to clang
                      Fix(Test.range("insert"), "#include \"x.h\"\n",
                          "Add include \"x.h\" for symbol clang::clangd::X"))),
          AllOf(
              Diag(Test.range("x"), "no type named 'X' in namespace 'clang'"),
              WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                          "Add include \"x.h\" for symbol clang::clangd::X"))),
          AllOf(
              Diag(Test.range("q2"), "use of undeclared identifier 'clangd'; "
                                     "did you mean 'clang'?"),
              WithFix(
                  _, // change clangd to clangd
                  Fix(Test.range("insert"), "#include \"y.h\"\n",
                      "Add include \"y.h\" for symbol clang::clangd::ns::Y"))),
          AllOf(Diag(Test.range("ns"),
                     "no member named 'ns' in namespace 'clang'"),
                WithFix(Fix(
                    Test.range("insert"), "#include \"y.h\"\n",
                    "Add include \"y.h\" for symbol clang::clangd::ns::Y")))));
}

TEST(IncludeFixerTest, SpecifiedScopeIsNamespaceAlias) {
  Annotations Test(R"cpp(
$insert[[]]namespace a {}
namespace b = a;
namespace c {
  b::$[[X]] x;
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"a::X", "unittest:///x.h", "\"x.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Test.range(), "no type named 'X' in namespace 'a'"),
                  WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                              "Add include \"x.h\" for symbol a::X")))));
}

TEST(DiagsInHeaders, DiagInsideHeader) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  Annotations Header("[[no_type_spec]];");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", Header.code()}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations"),
                  WithNote(Diag(Header.range(), "error occurred here")))));
}

TEST(DiagsInHeaders, DiagInTransitiveInclude) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"b.h\""}, {"b.h", "no_type_spec;"}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, DiagInMultipleHeaders) {
  Annotations Main(R"cpp(
    #include $a[["a.h"]]
    #include $b[["b.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "no_type_spec;"}, {"b.h", "no_type_spec;"}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range("a"), "in included file: C++ requires a type "
                                        "specifier for all declarations"),
                  Diag(Main.range("b"), "in included file: C++ requires a type "
                                        "specifier for all declarations")));
}

TEST(DiagsInHeaders, PreferExpansionLocation) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    #include "b.h"
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"b.h\"\n"},
                        {"b.h", "#ifndef X\n#define X\nno_type_spec;\n#endif"}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(Diag(Main.range(),
                                        "in included file: C++ requires a type "
                                        "specifier for all declarations")));
}

TEST(DiagsInHeaders, PreferExpansionLocationMacros) {
  Annotations Main(R"cpp(
    #define X
    #include "a.h"
    #undef X
    #include [["b.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"c.h\"\n"},
                        {"b.h", "#include \"c.h\"\n"},
                        {"c.h", "#ifndef X\n#define X\nno_type_spec;\n#endif"}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, LimitDiagsOutsideMainFile) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    #include "b.h"
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"c.h\"\n"},
                        {"b.h", "#include \"c.h\"\n"},
                        {"c.h", R"cpp(
      #ifndef X
      #define X
      no_type_spec_0;
      no_type_spec_1;
      no_type_spec_2;
      no_type_spec_3;
      no_type_spec_4;
      no_type_spec_5;
      no_type_spec_6;
      no_type_spec_7;
      no_type_spec_8;
      no_type_spec_9;
      no_type_spec_10;
      #endif)cpp"}};
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, OnlyErrorOrFatal) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  Annotations Header(R"cpp(
    [[no_type_spec]];
    int x = 5/0;)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", Header.code()}};
  auto diags = TU.build().getDiagnostics();
  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Main.range(), "in included file: C++ requires "
                                     "a type specifier for all declarations"),
                  WithNote(Diag(Header.range(), "error occurred here")))));
}
} // namespace

} // namespace clangd
} // namespace clang
