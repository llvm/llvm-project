//===- TreeTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Tree.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Mutations.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace clang;

namespace {
static llvm::ArrayRef<syntax::Token> tokens(syntax::Node *N) {
  assert(N->isOriginal() && "tokens of modified nodes are not well-defined");
  if (auto *L = dyn_cast<syntax::Leaf>(N))
    return llvm::makeArrayRef(L->token(), 1);
  auto *T = cast<syntax::Tree>(N);
  return llvm::makeArrayRef(T->firstLeaf()->token(),
                            T->lastLeaf()->token() + 1);
}

struct TestClangConfig {
  TestLanguage Language;
  std::string Target;

  bool isC99OrLater() const { return Language == Lang_C99; }

  bool isCXX() const {
    return Language == Lang_CXX03 || Language == Lang_CXX11 ||
           Language == Lang_CXX14 || Language == Lang_CXX17 ||
           Language == Lang_CXX20;
  }

  bool isCXX11OrLater() const {
    return Language == Lang_CXX11 || Language == Lang_CXX14 ||
           Language == Lang_CXX17 || Language == Lang_CXX20;
  }

  bool supportsCXXDynamicExceptionSpecification() const {
    return Language == Lang_CXX03 || Language == Lang_CXX11 ||
           Language == Lang_CXX14;
  }

  bool hasDelayedTemplateParsing() const {
    return Target == "x86_64-pc-win32-msvc";
  }

  std::vector<std::string> getCommandLineArgs() const {
    std::vector<std::string> Result = getCommandLineArgsForTesting(Language);
    Result.push_back("-target");
    Result.push_back(Target);
    return Result;
  }

  std::string toString() const {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    OS << "{ Language=" << Language << ", Target=" << Target << " }";
    return OS.str();
  }

  friend std::ostream &operator<<(std::ostream &OS,
                                  const TestClangConfig &ClangConfig) {
    return OS << ClangConfig.toString();
  }

  static std::vector<TestClangConfig> &allConfigs() {
    static std::vector<TestClangConfig> all_configs = []() {
      std::vector<TestClangConfig> all_configs;
      for (TestLanguage lang : {Lang_C89, Lang_C99, Lang_CXX03, Lang_CXX11,
                                Lang_CXX14, Lang_CXX17, Lang_CXX20}) {
        TestClangConfig config;
        config.Language = lang;
        config.Target = "x86_64-pc-linux-gnu";
        all_configs.push_back(config);

        // Windows target is interesting to test because it enables
        // `-fdelayed-template-parsing`.
        config.Target = "x86_64-pc-win32-msvc";
        all_configs.push_back(config);
      }
      return all_configs;
    }();
    return all_configs;
  }
};

class SyntaxTreeTest : public ::testing::Test,
                       public ::testing::WithParamInterface<TestClangConfig> {
protected:
  // Build a syntax tree for the code.
  syntax::TranslationUnit *buildTree(llvm::StringRef Code,
                                     const TestClangConfig &ClangConfig) {
    // FIXME: this code is almost the identical to the one in TokensTest. Share
    //        it.
    class BuildSyntaxTree : public ASTConsumer {
    public:
      BuildSyntaxTree(syntax::TranslationUnit *&Root,
                      std::unique_ptr<syntax::Arena> &Arena,
                      std::unique_ptr<syntax::TokenCollector> Tokens)
          : Root(Root), Arena(Arena), Tokens(std::move(Tokens)) {
        assert(this->Tokens);
      }

      void HandleTranslationUnit(ASTContext &Ctx) override {
        Arena = std::make_unique<syntax::Arena>(Ctx.getSourceManager(),
                                                Ctx.getLangOpts(),
                                                std::move(*Tokens).consume());
        Tokens = nullptr; // make sure we fail if this gets called twice.
        Root = syntax::buildSyntaxTree(*Arena, *Ctx.getTranslationUnitDecl());
      }

    private:
      syntax::TranslationUnit *&Root;
      std::unique_ptr<syntax::Arena> &Arena;
      std::unique_ptr<syntax::TokenCollector> Tokens;
    };

    class BuildSyntaxTreeAction : public ASTFrontendAction {
    public:
      BuildSyntaxTreeAction(syntax::TranslationUnit *&Root,
                            std::unique_ptr<syntax::Arena> &Arena)
          : Root(Root), Arena(Arena) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        // We start recording the tokens, ast consumer will take on the result.
        auto Tokens =
            std::make_unique<syntax::TokenCollector>(CI.getPreprocessor());
        return std::make_unique<BuildSyntaxTree>(Root, Arena,
                                                 std::move(Tokens));
      }

    private:
      syntax::TranslationUnit *&Root;
      std::unique_ptr<syntax::Arena> &Arena;
    };

    constexpr const char *FileName = "./input.cpp";
    FS->addFile(FileName, time_t(), llvm::MemoryBuffer::getMemBufferCopy(""));

    if (!Diags->getClient())
      Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), DiagOpts.get()));
    Diags->setSeverityForGroup(diag::Flavor::WarningOrError, "unused-value",
                               diag::Severity::Ignored, SourceLocation());

    // Prepare to run a compiler.
    std::vector<std::string> Args = {
        "syntax-test",
        "-fsyntax-only",
    };
    llvm::copy(ClangConfig.getCommandLineArgs(), std::back_inserter(Args));
    Args.push_back(FileName);

    std::vector<const char *> ArgsCStr;
    for (const std::string &arg : Args) {
      ArgsCStr.push_back(arg.c_str());
    }

    Invocation = createInvocationFromCommandLine(ArgsCStr, Diags, FS);
    assert(Invocation);
    Invocation->getFrontendOpts().DisableFree = false;
    Invocation->getPreprocessorOpts().addRemappedFile(
        FileName, llvm::MemoryBuffer::getMemBufferCopy(Code).release());
    CompilerInstance Compiler;
    Compiler.setInvocation(Invocation);
    Compiler.setDiagnostics(Diags.get());
    Compiler.setFileManager(FileMgr.get());
    Compiler.setSourceManager(SourceMgr.get());

    syntax::TranslationUnit *Root = nullptr;
    BuildSyntaxTreeAction Recorder(Root, this->Arena);

    // Action could not be executed but the frontend didn't identify any errors
    // in the code ==> problem in setting up the action.
    if (!Compiler.ExecuteAction(Recorder) &&
        Diags->getClient()->getNumErrors() == 0) {
      ADD_FAILURE() << "failed to run the frontend";
      std::abort();
    }
    return Root;
  }

  void expectTreeDumpEqual(StringRef Code, StringRef Tree) {
    SCOPED_TRACE(llvm::join(GetParam().getCommandLineArgs(), " "));
    SCOPED_TRACE(Code);

    auto *Root = buildTree(Code, GetParam());
    EXPECT_EQ(Diags->getClient()->getNumErrors(), 0u)
        << "Source file has syntax errors, they were printed to the test log";
    std::string Actual = std::string(StringRef(Root->dump(*Arena)).trim());
    EXPECT_EQ(Tree.trim().str(), Actual);
  }

  // Adds a file to the test VFS.
  void addFile(llvm::StringRef Path, llvm::StringRef Contents) {
    if (!FS->addFile(Path, time_t(),
                     llvm::MemoryBuffer::getMemBufferCopy(Contents))) {
      ADD_FAILURE() << "could not add a file to VFS: " << Path;
    }
  }

  /// Finds the deepest node in the tree that covers exactly \p R.
  /// FIXME: implement this efficiently and move to public syntax tree API.
  syntax::Node *nodeByRange(llvm::Annotations::Range R, syntax::Node *Root) {
    llvm::ArrayRef<syntax::Token> Toks = tokens(Root);

    if (Toks.front().location().isFileID() &&
        Toks.back().location().isFileID() &&
        syntax::Token::range(*SourceMgr, Toks.front(), Toks.back()) ==
            syntax::FileRange(SourceMgr->getMainFileID(), R.Begin, R.End))
      return Root;

    auto *T = dyn_cast<syntax::Tree>(Root);
    if (!T)
      return nullptr;
    for (auto *C = T->firstChild(); C != nullptr; C = C->nextSibling()) {
      if (auto *Result = nodeByRange(R, C))
        return Result;
    }
    return nullptr;
  }

  // Data fields.
  llvm::IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      new DiagnosticOptions();
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      new DiagnosticsEngine(new DiagnosticIDs, DiagOpts.get());
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      new llvm::vfs::InMemoryFileSystem;
  llvm::IntrusiveRefCntPtr<FileManager> FileMgr =
      new FileManager(FileSystemOptions(), FS);
  llvm::IntrusiveRefCntPtr<SourceManager> SourceMgr =
      new SourceManager(*Diags, *FileMgr);
  std::shared_ptr<CompilerInvocation> Invocation;
  // Set after calling buildTree().
  std::unique_ptr<syntax::Arena> Arena;
};

TEST_P(SyntaxTreeTest, Simple) {
  expectTreeDumpEqual(
      R"cpp(
int main() {}
void foo() {}
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-main
| | `-ParametersAndQualifiers
| |   |-(
| |   `-)
| `-CompoundStatement
|   |-{
|   `-}
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, SimpleVariable) {
  expectTreeDumpEqual(
      R"cpp(
int a;
int b = 42;
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | `-a
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-b
  | |-=
  | `-UnknownExpression
  |   `-42
  `-;
)txt");
}

TEST_P(SyntaxTreeTest, SimpleFunction) {
  expectTreeDumpEqual(
      R"cpp(
void foo(int a, int b) {}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-b
  |   `-)
  `-CompoundStatement
    |-{
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, If) {
  expectTreeDumpEqual(
      R"cpp(
int main() {
  if (1) {}
  if (1) {} else if (0) {}
}
        )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-main
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-1
    | |-)
    | |-CompoundStatement
    | | |-{
    | | `-}
    | |-else
    | `-IfStatement
    |   |-if
    |   |-(
    |   |-UnknownExpression
    |   | `-0
    |   |-)
    |   `-CompoundStatement
    |     |-{
    |     `-}
    `-}
        )txt");
}

TEST_P(SyntaxTreeTest, For) {
  expectTreeDumpEqual(
      R"cpp(
void test() {
  for (;;)  {}
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-ForStatement
    | |-for
    | |-(
    | |-;
    | |-;
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    `-}
        )txt");
}

TEST_P(SyntaxTreeTest, RangeBasedFor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
void test() {
  int a[3];
  for (int x : a)
    ;
}
      )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | `-SimpleDeclarator
    | |   |-a
    | |   `-ArraySubscript
    | |     |-[
    | |     |-UnknownExpression
    | |     | `-3
    | |     `-]
    | `-;
    |-RangeBasedForStatement
    | |-for
    | |-(
    | |-SimpleDeclaration
    | | |-int
    | | |-SimpleDeclarator
    | | | `-x
    | | `-:
    | |-UnknownExpression
    | | `-a
    | |-)
    | `-EmptyStatement
    |   `-;
    `-}
       )txt");
}

TEST_P(SyntaxTreeTest, DeclarationStatement) {
  expectTreeDumpEqual("void test() { int a = 10; }",
                      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | `-SimpleDeclarator
    | |   |-a
    | |   |-=
    | |   `-UnknownExpression
    | |     `-10
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, Switch) {
  expectTreeDumpEqual(
      R"cpp(
void test() {
  switch (1) {
    case 0:
    default:;
  }
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-SwitchStatement
    | |-switch
    | |-(
    | |-UnknownExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-CaseStatement
    |   | |-case
    |   | |-UnknownExpression
    |   | | `-0
    |   | |-:
    |   | `-DefaultStatement
    |   |   |-default
    |   |   |-:
    |   |   `-EmptyStatement
    |   |     `-;
    |   `-}
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, While) {
  expectTreeDumpEqual(
      R"cpp(
void test() {
  while (1) { continue; break; }
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-WhileStatement
    | |-while
    | |-(
    | |-UnknownExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-ContinueStatement
    |   | |-continue
    |   | `-;
    |   |-BreakStatement
    |   | |-break
    |   | `-;
    |   `-}
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, UnhandledStatement) {
  // Unhandled statements should end up as 'unknown statement'.
  // This example uses a 'label statement', which does not yet have a syntax
  // counterpart.
  expectTreeDumpEqual("int main() { foo: return 100; }",
                      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-main
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-UnknownStatement
    | |-foo
    | |-:
    | `-ReturnStatement
    |   |-return
    |   |-UnknownExpression
    |   | `-100
    |   `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, Expressions) {
  // expressions should be wrapped in 'ExpressionStatement' when they appear
  // in a statement position.
  expectTreeDumpEqual(
      R"cpp(
void test() {
  test();
  if (1) test(); else test();
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-UnknownExpression
    | | | `-test
    | | |-(
    | | `-)
    | `-;
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-1
    | |-)
    | |-ExpressionStatement
    | | |-UnknownExpression
    | | | |-UnknownExpression
    | | | | `-test
    | | | |-(
    | | | `-)
    | | `-;
    | |-else
    | `-ExpressionStatement
    |   |-UnknownExpression
    |   | |-UnknownExpression
    |   | | `-test
    |   | |-(
    |   | `-)
    |   `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, CxxNullPtrLiteral) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
void test() {
  nullptr;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-CxxNullPtrExpression
    | | `-nullptr
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, PostfixUnaryOperator) {
  expectTreeDumpEqual(
      R"cpp(
void test(int a) {
  a++;
  a--;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PostfixUnaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | `-++
    | `-;
    |-ExpressionStatement
    | |-PostfixUnaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | `---
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperator) {
  expectTreeDumpEqual(
      R"cpp(
void test(int a, int *ap) {
  --a; ++a;
  ~a;
  -a;
  +a;
  &a;
  *ap;
  !a;
  __real a; __imag a;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-*
  |   |   `-ap
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |---
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-++
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-~
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |--
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-+
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-&
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-*
    | | `-UnknownExpression
    | |   `-ap
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-!
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-__real
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-__imag
    | | `-UnknownExpression
    | |   `-a
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
void test(int a, bool b) {
  compl a;
  not b;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-bool
  |   | `-SimpleDeclarator
  |   |   `-b
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-compl
    | | `-UnknownExpression
    | |   `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-not
    | | `-UnknownExpression
    | |   `-b
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, BinaryOperator) {
  expectTreeDumpEqual(
      R"cpp(
void test(int a) {
  1 - 2;
  1 == 2;
  a = 1;
  a <<= 1;
  1 || 0;
  1 & 2;
  a ^= 3;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-1
    | | |--
    | | `-UnknownExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-1
    | | |-==
    | | `-UnknownExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | |-=
    | | `-UnknownExpression
    | |   `-1
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | |-<<=
    | | `-UnknownExpression
    | |   `-1
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-1
    | | |-||
    | | `-UnknownExpression
    | |   `-0
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-1
    | | |-&
    | | `-UnknownExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | |-^=
    | | `-UnknownExpression
    | |   `-3
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, BinaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
void test(int a) {
  true || false;
  true or false;
  1 bitand 2;
  a xor_eq 3;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-true
    | | |-||
    | | `-UnknownExpression
    | |   `-false
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-true
    | | |-or
    | | `-UnknownExpression
    | |   `-false
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-1
    | | |-bitand
    | | `-UnknownExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | |-xor_eq
    | | `-UnknownExpression
    | |   `-3
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, NestedBinaryOperator) {
  expectTreeDumpEqual(
      R"cpp(
void test(int a, int b) {
  (1 + 2) * (4 / 2);
  a + b + 42;
  a = b = 42;
  a + b * 4 + 2;
  a % 2 + b * 42;
}
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-b
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | |-(
    | | | |-BinaryOperatorExpression
    | | | | |-UnknownExpression
    | | | | | `-1
    | | | | |-+
    | | | | `-UnknownExpression
    | | | |   `-2
    | | | `-)
    | | |-*
    | | `-UnknownExpression
    | |   |-(
    | |   |-BinaryOperatorExpression
    | |   | |-UnknownExpression
    | |   | | `-4
    | |   | |-/
    | |   | `-UnknownExpression
    | |   |   `-2
    | |   `-)
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-UnknownExpression
    | | | | `-a
    | | | |-+
    | | | `-UnknownExpression
    | | |   `-b
    | | |-+
    | | `-UnknownExpression
    | |   `-42
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-a
    | | |-=
    | | `-BinaryOperatorExpression
    | |   |-UnknownExpression
    | |   | `-b
    | |   |-=
    | |   `-UnknownExpression
    | |     `-42
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-UnknownExpression
    | | | | `-a
    | | | |-+
    | | | `-BinaryOperatorExpression
    | | |   |-UnknownExpression
    | | |   | `-b
    | | |   |-*
    | | |   `-UnknownExpression
    | | |     `-4
    | | |-+
    | | `-UnknownExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-UnknownExpression
    | | | | `-a
    | | | |-%
    | | | `-UnknownExpression
    | | |   `-2
    | | |-+
    | | `-BinaryOperatorExpression
    | |   |-UnknownExpression
    | |   | `-b
    | |   |-*
    | |   `-UnknownExpression
    | |     `-42
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, UserDefinedBinaryOperator) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
struct X {
  X& operator=(const X&);
  friend X operator+(X, const X&);
  friend bool operator<(const X&, const X&);
};
void test(X x, X y) {
  x = y;
  x + y;
  x < y;
}
      )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-&
| | | |-operator
| | | |-=
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   |-SimpleDeclaration
| | |   | |-const
| | |   | |-X
| | |   | `-SimpleDeclarator
| | |   |   `-&
| | |   `-)
| | `-;
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-X
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-+
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | `-X
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-bool
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-<
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-x
    | | |-UnknownExpression
    | | | `-=
    | | `-UnknownExpression
    | |   `-y
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-UnknownExpression
    | | |   `-x
    | | |-UnknownExpression
    | | | `-+
    | | `-UnknownExpression
    | |   `-y
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-x
    | | |-UnknownExpression
    | | | `-<
    | | `-UnknownExpression
    | |   `-y
    | `-;
    `-}
)txt");
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGrouping) {
  expectTreeDumpEqual(
      R"cpp(
      int *a, b; int *c, d;
  )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-*
| | `-a
| |-,
| |-SimpleDeclarator
| | `-b
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-*
  | `-c
  |-,
  |-SimpleDeclarator
  | `-d
  `-;
  )txt");
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGroupingTypedef) {
  expectTreeDumpEqual(
      R"cpp(
    typedef int *a, b;
  )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-typedef
  |-int
  |-SimpleDeclarator
  | |-*
  | `-a
  |-,
  |-SimpleDeclarator
  | `-b
  `-;
  )txt");
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsInsideStatement) {
  expectTreeDumpEqual(
      R"cpp(
void foo() {
      int *a, b;
      typedef int *ta, tb;
}
  )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | |-SimpleDeclarator
    | | | |-*
    | | | `-a
    | | |-,
    | | `-SimpleDeclarator
    | |   `-b
    | `-;
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-typedef
    | | |-int
    | | |-SimpleDeclarator
    | | | |-*
    | | | `-ta
    | | |-,
    | | `-SimpleDeclarator
    | |   `-tb
    | `-;
    `-}
  )txt");
}

TEST_P(SyntaxTreeTest, Namespaces) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
namespace a { namespace b {} }
namespace a::b {}
namespace {}

namespace foo = a;
    )cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-a
| |-{
| |-NamespaceDefinition
| | |-namespace
| | |-b
| | |-{
| | `-}
| `-}
|-NamespaceDefinition
| |-namespace
| |-a
| |-::
| |-b
| |-{
| `-}
|-NamespaceDefinition
| |-namespace
| |-{
| `-}
`-NamespaceAliasDefinition
  |-namespace
  |-foo
  |-=
  |-a
  `-;
)txt");
}

TEST_P(SyntaxTreeTest, UsingDirective) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
namespace ns {}
using namespace ::ns;
    )cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| `-}
`-UsingNamespaceDirective
  |-using
  |-namespace
  |-::
  |-ns
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, UsingDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
namespace ns { int a; }
using ns::a;
    )cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| |-SimpleDeclaration
| | |-int
| | |-SimpleDeclarator
| | | `-a
| | `-;
| `-}
`-UsingDeclaration
  |-using
  |-ns
  |-::
  |-a
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, FreeStandingClasses) {
  // Free-standing classes, must live inside a SimpleDeclaration.
  expectTreeDumpEqual(
      R"cpp(
struct X;
struct X {};

struct Y *y1;
struct Y {} *y2;

struct {} *a1;
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| `-;
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-struct
| |-Y
| |-SimpleDeclarator
| | |-*
| | `-y1
| `-;
|-SimpleDeclaration
| |-struct
| |-Y
| |-{
| |-}
| |-SimpleDeclarator
| | |-*
| | `-y2
| `-;
`-SimpleDeclaration
  |-struct
  |-{
  |-}
  |-SimpleDeclarator
  | |-*
  | `-a1
  `-;
)txt");
}

TEST_P(SyntaxTreeTest, Templates) {
  if (!GetParam().isCXX()) {
    return;
  }
  if (GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Make this test work on Windows by generating the expected syntax
    // tree when `-fdelayed-template-parsing` is active.
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
template <class T> struct cls {};
template <class T> int var = 10;
template <class T> int fun() {}
    )cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-cls
|   |-{
|   |-}
|   `-;
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-int
|   |-SimpleDeclarator
|   | |-var
|   | |-=
|   | `-UnknownExpression
|   |   `-10
|   `-;
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-int
    |-SimpleDeclarator
    | |-fun
    | `-ParametersAndQualifiers
    |   |-(
    |   `-)
    `-CompoundStatement
      |-{
      `-}
)txt");
}

TEST_P(SyntaxTreeTest, NestedTemplates) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
template <class T>
struct X {
  template <class U>
  U foo();
};
    )cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-X
    |-{
    |-TemplateDeclaration
    | |-template
    | |-<
    | |-UnknownDeclaration
    | | |-class
    | | `-U
    | |->
    | `-SimpleDeclaration
    |   |-U
    |   |-SimpleDeclarator
    |   | |-foo
    |   | `-ParametersAndQualifiers
    |   |   |-(
    |   |   `-)
    |   `-;
    |-}
    `-;
)txt");
}

TEST_P(SyntaxTreeTest, Templates2) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
template <class T> struct X { struct Y; };
template <class T> struct X<T>::Y {};
    )cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-{
|   |-SimpleDeclaration
|   | |-struct
|   | |-Y
|   | `-;
|   |-}
|   `-;
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-X
    |-<
    |-T
    |->
    |-::
    |-Y
    |-{
    |-}
    `-;
       )txt");
}

TEST_P(SyntaxTreeTest, TemplatesUsingUsing) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
template <class T> struct X {
  using T::foo;
  using typename T::bar;
};
    )cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-X
    |-{
    |-UsingDeclaration
    | |-using
    | |-T
    | |-::
    | |-foo
    | `-;
    |-UsingDeclaration
    | |-using
    | |-typename
    | |-T
    | |-::
    | |-bar
    | `-;
    |-}
    `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ExplicitTemplateInstantations) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
template <class T> struct X {};
template <class T> struct X<T*> {};
template <> struct X<int> {};

template struct X<double>;
extern template struct X<float>;
)cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-{
|   |-}
|   `-;
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-T
|   |-*
|   |->
|   |-{
|   |-}
|   `-;
|-TemplateDeclaration
| |-template
| |-<
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-int
|   |->
|   |-{
|   |-}
|   `-;
|-ExplicitTemplateInstantiation
| |-template
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-double
|   |->
|   `-;
`-ExplicitTemplateInstantiation
  |-extern
  |-template
  `-SimpleDeclaration
    |-struct
    |-X
    |-<
    |-float
    |->
    `-;
)txt");
}

TEST_P(SyntaxTreeTest, UsingType) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
using type = int;
    )cpp",
      R"txt(
*: TranslationUnit
`-TypeAliasDeclaration
  |-using
  |-type
  |-=
  |-int
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, EmptyDeclaration) {
  expectTreeDumpEqual(
      R"cpp(
;
    )cpp",
      R"txt(
*: TranslationUnit
`-EmptyDeclaration
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, StaticAssert) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
static_assert(true, "message");
static_assert(true);
    )cpp",
      R"txt(
*: TranslationUnit
|-StaticAssertDeclaration
| |-static_assert
| |-(
| |-UnknownExpression
| | `-true
| |-,
| |-UnknownExpression
| | `-"message"
| |-)
| `-;
`-StaticAssertDeclaration
  |-static_assert
  |-(
  |-UnknownExpression
  | `-true
  |-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ExternC) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
extern "C" int a;
extern "C" { int b; int c; }
    )cpp",
      R"txt(
*: TranslationUnit
|-LinkageSpecificationDeclaration
| |-extern
| |-"C"
| `-SimpleDeclaration
|   |-int
|   |-SimpleDeclarator
|   | `-a
|   `-;
`-LinkageSpecificationDeclaration
  |-extern
  |-"C"
  |-{
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | `-b
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | `-c
  | `-;
  `-}
       )txt");
}

TEST_P(SyntaxTreeTest, NonModifiableNodes) {
  // Some nodes are non-modifiable, they are marked with 'I:'.
  expectTreeDumpEqual(
      R"cpp(
#define HALF_IF if (1+
#define HALF_IF_2 1) {}
void test() {
  HALF_IF HALF_IF_2 else {}
})cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-IfStatement
    | |-I: if
    | |-I: (
    | |-I: BinaryOperatorExpression
    | | |-I: UnknownExpression
    | | | `-I: 1
    | | |-I: +
    | | `-I: UnknownExpression
    | |   `-I: 1
    | |-I: )
    | |-I: CompoundStatement
    | | |-I: {
    | | `-I: }
    | |-else
    | `-CompoundStatement
    |   |-{
    |   `-}
    `-}
       )txt");
}

TEST_P(SyntaxTreeTest, ModifiableNodes) {
  // All nodes can be mutated.
  expectTreeDumpEqual(
      R"cpp(
#define OPEN {
#define CLOSE }

void test() {
  OPEN
    1;
  CLOSE

  OPEN
    2;
  }
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-CompoundStatement
    | |-{
    | |-ExpressionStatement
    | | |-UnknownExpression
    | | | `-1
    | | `-;
    | `-}
    |-CompoundStatement
    | |-{
    | |-ExpressionStatement
    | | |-UnknownExpression
    | | | `-2
    | | `-;
    | `-}
    `-}
       )txt");
}

TEST_P(SyntaxTreeTest, ArraySubscriptsInDeclarators) {
  expectTreeDumpEqual(
      R"cpp(
int a[10];
int b[1][2][3];
int c[] = {1,2,3};
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ArraySubscript
| |   |-[
| |   |-UnknownExpression
| |   | `-10
| |   `-]
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-b
| | |-ArraySubscript
| | | |-[
| | | |-UnknownExpression
| | | | `-1
| | | `-]
| | |-ArraySubscript
| | | |-[
| | | |-UnknownExpression
| | | | `-2
| | | `-]
| | `-ArraySubscript
| |   |-[
| |   |-UnknownExpression
| |   | `-3
| |   `-]
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-c
  | |-ArraySubscript
  | | |-[
  | | `-]
  | |-=
  | `-UnknownExpression
  |   `-UnknownExpression
  |     |-{
  |     |-UnknownExpression
  |     | `-1
  |     |-,
  |     |-UnknownExpression
  |     | `-2
  |     |-,
  |     |-UnknownExpression
  |     | `-3
  |     `-}
  `-;       )txt");
}

TEST_P(SyntaxTreeTest, StaticArraySubscriptsInDeclarators) {
  if (!GetParam().isC99OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
void f(int xs[static 10]);
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-f
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-xs
  |   |   `-ArraySubscript
  |   |     |-[
  |   |     |-static
  |   |     |-UnknownExpression
  |   |     | `-10
  |   |     `-]
  |   `-)
  `-;       )txt");
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctions) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
int func1();
int func2a(int a);
int func2b(int);
int func3a(int *ap);
int func3b(int *);
int func4a(int a, float b);
int func4b(int, float);
      )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func1
| | `-ParametersAndQualifiers
| |   |-(
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func2a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func2b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-int
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func3a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   |-*
| |   |   `-ap
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func3b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-*
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func4a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   |-,
| |   |-SimpleDeclaration
| |   | |-float
| |   | `-SimpleDeclarator
| |   |   `-b
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func4b
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | `-int
  |   |-,
  |   |-SimpleDeclaration
  |   | `-float
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctionsCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
int func1(const int a, volatile int b, const volatile int c);
int func2(int& a);
      )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func1
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   |-,
| |   |-SimpleDeclaration
| |   | |-volatile
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-b
| |   |-,
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-volatile
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-c
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func2
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-&
  |   |   `-a
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctionsCxx11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
int func1(int&& a);
      )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func1
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-&&
  |   |   `-a
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInMemberFunctions) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
struct Test {
  int a();
  int b() const;
  int c() volatile;
  int d() const volatile;
  int e() &;
  int f() &&;
};
      )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-Test
  |-{
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-a
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   `-)
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-b
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-const
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-c
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-volatile
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-d
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   |-const
  | |   `-volatile
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-e
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-&
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-f
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-&&
  | `-;
  |-}
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, TrailingReturn) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
auto foo() -> int;
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-auto
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   `-TrailingReturnType
  |     |-->
  |     `-int
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, DynamicExceptionSpecification) {
  if (!GetParam().supportsCXXDynamicExceptionSpecification()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
struct MyException1 {};
struct MyException2 {};
int a() throw();
int b() throw(...);
int c() throw(MyException1);
int d() throw(MyException1, MyException2);
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-MyException1
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-struct
| |-MyException2
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   |-...
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-c
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   |-MyException1
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-d
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   |-throw
  |   |-(
  |   |-MyException1
  |   |-,
  |   |-MyException2
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, NoexceptExceptionSpecification) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
int a() noexcept;
int b() noexcept(true);
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   `-noexcept
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-b
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   |-noexcept
  |   |-(
  |   |-UnknownExpression
  |   | `-true
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, DeclaratorsInParentheses) {
  expectTreeDumpEqual(
      R"cpp(
int (a);
int *(b);
int (*c)(int);
int *(d)(int);
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | `-ParenDeclarator
| |   |-(
| |   |-a
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-*
| | `-ParenDeclarator
| |   |-(
| |   |-b
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-ParenDeclarator
| | | |-(
| | | |-*
| | | |-c
| | | `-)
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-int
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-*
  | |-ParenDeclarator
  | | |-(
  | | |-d
  | | `-)
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | `-int
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ConstVolatileQualifiers) {
  expectTreeDumpEqual(
      R"cpp(
const int west = -1;
int const east = 1;
const int const universal = 0;
const int const *const *volatile b;
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-const
| |-int
| |-SimpleDeclarator
| | |-west
| | |-=
| | `-PrefixUnaryOperatorExpression
| |   |--
| |   `-UnknownExpression
| |     `-1
| `-;
|-SimpleDeclaration
| |-int
| |-const
| |-SimpleDeclarator
| | |-east
| | |-=
| | `-UnknownExpression
| |   `-1
| `-;
|-SimpleDeclaration
| |-const
| |-int
| |-const
| |-SimpleDeclarator
| | |-universal
| | |-=
| | `-UnknownExpression
| |   `-0
| `-;
`-SimpleDeclaration
  |-const
  |-int
  |-const
  |-SimpleDeclarator
  | |-*
  | |-const
  | |-*
  | |-volatile
  | `-b
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, RangesOfDeclaratorsWithTrailingReturnTypes) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
auto foo() -> auto(*)(int) -> double*;
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-auto
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   `-TrailingReturnType
  |     |-->
  |     |-auto
  |     `-SimpleDeclarator
  |       |-ParenDeclarator
  |       | |-(
  |       | |-*
  |       | `-)
  |       `-ParametersAndQualifiers
  |         |-(
  |         |-SimpleDeclaration
  |         | `-int
  |         |-)
  |         `-TrailingReturnType
  |           |-->
  |           |-double
  |           `-SimpleDeclarator
  |             `-*
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, MemberPointers) {
  if (!GetParam().isCXX()) {
    return;
  }
  expectTreeDumpEqual(
      R"cpp(
struct X {};
int X::* a;
const int X::* b;
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-MemberPointer
| | | |-X
| | | |-::
| | | `-*
| | `-a
| `-;
`-SimpleDeclaration
  |-const
  |-int
  |-SimpleDeclarator
  | |-MemberPointer
  | | |-X
  | | |-::
  | | `-*
  | `-b
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ComplexDeclarator) {
  expectTreeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int));
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-x
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-char
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-short
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-b
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | `-int
  |   |     `-)
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, ComplexDeclarator2) {
  expectTreeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int), long (**c)(long long));
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-x
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-char
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-short
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-b
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | `-int
  |   |     `-)
  |   |-,
  |   |-SimpleDeclaration
  |   | |-long
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-*
  |   |   | |-c
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | |-long
  |   |     | `-long
  |   |     `-)
  |   `-)
  `-;
       )txt");
}

TEST_P(SyntaxTreeTest, Mutations) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }

  using Transformation = std::function<void(
      const llvm::Annotations & /*Input*/, syntax::TranslationUnit * /*Root*/)>;
  auto CheckTransformation = [this](std::string Input, std::string Expected,
                                    Transformation Transform) -> void {
    llvm::Annotations Source(Input);
    auto *Root = buildTree(Source.code(), GetParam());

    Transform(Source, Root);

    auto Replacements = syntax::computeReplacements(*Arena, *Root);
    auto Output = tooling::applyAllReplacements(Source.code(), Replacements);
    if (!Output) {
      ADD_FAILURE() << "could not apply replacements: "
                    << llvm::toString(Output.takeError());
      return;
    }

    EXPECT_EQ(Expected, *Output) << "input is:\n" << Input;
  };

  // Removes the selected statement. Input should have exactly one selected
  // range and it should correspond to a single statement.
  auto RemoveStatement = [this](const llvm::Annotations &Input,
                                syntax::TranslationUnit *TU) {
    auto *S = cast<syntax::Statement>(nodeByRange(Input.range(), TU));
    ASSERT_TRUE(S->canModify()) << "cannot remove a statement";
    syntax::removeStatement(*Arena, S);
    EXPECT_TRUE(S->isDetached());
    EXPECT_FALSE(S->isOriginal())
        << "node removed from tree cannot be marked as original";
  };

  std::vector<std::pair<std::string /*Input*/, std::string /*Expected*/>>
      Cases = {
          {"void test() { [[100+100;]] test(); }", "void test() {  test(); }"},
          {"void test() { if (true) [[{}]] else {} }",
           "void test() { if (true) ; else {} }"},
          {"void test() { [[;]] }", "void test() {  }"}};
  for (const auto &C : Cases)
    CheckTransformation(C.first, C.second, RemoveStatement);
}

TEST_P(SyntaxTreeTest, SynthesizedNodes) {
  buildTree("", GetParam());

  auto *C = syntax::createPunctuation(*Arena, tok::comma);
  ASSERT_NE(C, nullptr);
  EXPECT_EQ(C->token()->kind(), tok::comma);
  EXPECT_TRUE(C->canModify());
  EXPECT_FALSE(C->isOriginal());
  EXPECT_TRUE(C->isDetached());

  auto *S = syntax::createEmptyStatement(*Arena);
  ASSERT_NE(S, nullptr);
  EXPECT_TRUE(S->canModify());
  EXPECT_FALSE(S->isOriginal());
  EXPECT_TRUE(S->isDetached());
}

INSTANTIATE_TEST_CASE_P(SyntaxTreeTests, SyntaxTreeTest,
                        testing::ValuesIn(TestClangConfig::allConfigs()));

} // namespace
