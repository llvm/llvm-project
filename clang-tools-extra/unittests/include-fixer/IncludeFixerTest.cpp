//===-- IncludeFixerTest.cpp - Include fixer unit tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemorySymbolIndex.h"
#include "IncludeFixer.h"
#include "SymbolIndexManager.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace include_fixer {
namespace {

using find_all_symbols::SymbolInfo;
using find_all_symbols::SymbolAndSignals;

static bool runOnCode(tooling::ToolAction *ToolAction, StringRef Code,
                      StringRef FileName,
                      const std::vector<std::string> &ExtraArgs) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), InMemoryFileSystem));
  // FIXME: Investigate why -fms-compatibility breaks tests.
  std::vector<std::string> Args = {"include_fixer", "-fsyntax-only",
                                   "-fno-ms-compatibility", FileName};
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  tooling::ToolInvocation Invocation(
      Args, ToolAction, Files.get(),
      std::make_shared<PCHContainerOperations>());

  InMemoryFileSystem->addFile(FileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(Code));

  InMemoryFileSystem->addFile("foo.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  InMemoryFileSystem->addFile("dir/bar.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  InMemoryFileSystem->addFile("dir/otherdir/qux.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  InMemoryFileSystem->addFile("header.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("bar b;"));
  return Invocation.run();
}

static std::string runIncludeFixer(
    StringRef Code,
    const std::vector<std::string> &ExtraArgs = std::vector<std::string>()) {
  std::vector<SymbolAndSignals> Symbols = {
      {SymbolInfo("string", SymbolInfo::SymbolKind::Class, "<string>",
                  {{SymbolInfo::ContextType::Namespace, "std"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("sting", SymbolInfo::SymbolKind::Class, "\"sting\"",
                  {{SymbolInfo::ContextType::Namespace, "std"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("foo", SymbolInfo::SymbolKind::Class,
                  "\"dir/otherdir/qux.h\"",
                  {{SymbolInfo::ContextType::Namespace, "b"},
                   {SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("bar", SymbolInfo::SymbolKind::Class, "\"bar.h\"",
                  {{SymbolInfo::ContextType::Namespace, "b"},
                   {SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("bar", SymbolInfo::SymbolKind::Class, "\"bar2.h\"",
                  {{SymbolInfo::ContextType::Namespace, "c"},
                   {SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("Green", SymbolInfo::SymbolKind::Class, "\"color.h\"",
                  {{SymbolInfo::ContextType::EnumDecl, "Color"},
                   {SymbolInfo::ContextType::Namespace, "b"},
                   {SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("Vector", SymbolInfo::SymbolKind::Class, "\"Vector.h\"",
                  {{SymbolInfo::ContextType::Namespace, "__a"},
                   {SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{/*Seen=*/2, 0}},
      {SymbolInfo("Vector", SymbolInfo::SymbolKind::Class, "\"Vector.h\"",
                  {{SymbolInfo::ContextType::Namespace, "a"}}),
       SymbolInfo::Signals{/*Seen=*/2, 0}},
      {SymbolInfo("StrCat", SymbolInfo::SymbolKind::Class, "\"strcat.h\"",
                  {{SymbolInfo::ContextType::Namespace, "str"}}),
       SymbolInfo::Signals{}},
      {SymbolInfo("str", SymbolInfo::SymbolKind::Class, "\"str.h\"", {}),
       SymbolInfo::Signals{}},
      {SymbolInfo("foo2", SymbolInfo::SymbolKind::Class, "\"foo2.h\"", {}),
       SymbolInfo::Signals{}},
  };
  auto SymbolIndexMgr = llvm::make_unique<SymbolIndexManager>();
  SymbolIndexMgr->addSymbolIndex(
      [=]() { return llvm::make_unique<InMemorySymbolIndex>(Symbols); });

  std::vector<IncludeFixerContext> FixerContexts;
  IncludeFixerActionFactory Factory(*SymbolIndexMgr, FixerContexts, "llvm");
  std::string FakeFileName = "input.cc";
  runOnCode(&Factory, Code, FakeFileName, ExtraArgs);
  assert(FixerContexts.size() == 1);
  if (FixerContexts.front().getHeaderInfos().empty())
    return Code;
  auto Replaces = createIncludeFixerReplacements(Code, FixerContexts.front());
  EXPECT_TRUE(static_cast<bool>(Replaces))
      << llvm::toString(Replaces.takeError()) << "\n";
  if (!Replaces)
    return "";
  RewriterTestContext Context;
  FileID ID = Context.createInMemoryFile(FakeFileName, Code);
  tooling::applyAllReplacements(*Replaces, Context.Rewrite);
  return Context.getRewrittenText(ID);
}

TEST(IncludeFixer, Typo) {
  EXPECT_EQ("#include <string>\nstd::string foo;\n",
            runIncludeFixer("std::string foo;\n"));

  EXPECT_EQ("// comment\n#include \"foo.h\"\n#include <string>\n"
            "std::string foo;\n#include \"dir/bar.h\"\n",
            runIncludeFixer("// comment\n#include \"foo.h\"\nstd::string foo;\n"
                            "#include \"dir/bar.h\"\n"));

  EXPECT_EQ("#include \"foo.h\"\n#include <string>\nstd::string foo;\n",
            runIncludeFixer("#include \"foo.h\"\nstd::string foo;\n"));

  EXPECT_EQ(
      "#include \"foo.h\"\n#include <string>\nstd::string::size_type foo;\n",
      runIncludeFixer("#include \"foo.h\"\nstd::string::size_type foo;\n"));

  EXPECT_EQ("#include <string>\nstd::string foo;\n",
            runIncludeFixer("string foo;\n"));

  // Should not match std::string.
  EXPECT_EQ("::string foo;\n", runIncludeFixer("::string foo;\n"));
}

TEST(IncludeFixer, IncompleteType) {
  EXPECT_EQ(
      "#include \"foo.h\"\n#include <string>\n"
      "namespace std {\nclass string;\n}\nstd::string foo;\n",
      runIncludeFixer("#include \"foo.h\"\n"
                      "namespace std {\nclass string;\n}\nstring foo;\n"));

  EXPECT_EQ("#include <string>\n"
            "class string;\ntypedef string foo;\nfoo f;\n",
            runIncludeFixer("class string;\ntypedef string foo;\nfoo f;\n"));
}

TEST(IncludeFixer, MinimizeInclude) {
  std::vector<std::string> IncludePath = {"-Idir/"};
  EXPECT_EQ("#include \"otherdir/qux.h\"\na::b::foo bar;\n",
            runIncludeFixer("a::b::foo bar;\n", IncludePath));

  IncludePath = {"-isystemdir"};
  EXPECT_EQ("#include <otherdir/qux.h>\na::b::foo bar;\n",
            runIncludeFixer("a::b::foo bar;\n", IncludePath));

  IncludePath = {"-iquotedir"};
  EXPECT_EQ("#include \"otherdir/qux.h\"\na::b::foo bar;\n",
            runIncludeFixer("a::b::foo bar;\n", IncludePath));

  IncludePath = {"-Idir", "-Idir/otherdir"};
  EXPECT_EQ("#include \"qux.h\"\na::b::foo bar;\n",
            runIncludeFixer("a::b::foo bar;\n", IncludePath));
}

TEST(IncludeFixer, NestedName) {
  EXPECT_EQ("#include \"dir/otherdir/qux.h\"\n"
            "int x = a::b::foo(0);\n",
            runIncludeFixer("int x = a::b::foo(0);\n"));

  // FIXME: Handle simple macros.
  EXPECT_EQ("#define FOO a::b::foo\nint x = FOO;\n",
            runIncludeFixer("#define FOO a::b::foo\nint x = FOO;\n"));
  EXPECT_EQ("#define FOO(x) a::##x\nint x = FOO(b::foo);\n",
            runIncludeFixer("#define FOO(x) a::##x\nint x = FOO(b::foo);\n"));

  // The empty namespace is cleaned up by clang-format after include-fixer
  // finishes.
  EXPECT_EQ("#include \"dir/otherdir/qux.h\"\n"
            "\nint a = a::b::foo(0);\n",
            runIncludeFixer("namespace a {}\nint a = a::b::foo(0);\n"));
}

TEST(IncludeFixer, MultipleMissingSymbols) {
  EXPECT_EQ("#include <string>\nstd::string bar;\nstd::sting foo;\n",
            runIncludeFixer("std::string bar;\nstd::sting foo;\n"));
}

TEST(IncludeFixer, ScopedNamespaceSymbols) {
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nb::bar b;\n}",
            runIncludeFixer("namespace a {\nb::bar b;\n}"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace A {\na::b::bar b;\n}",
            runIncludeFixer("namespace A {\na::b::bar b;\n}"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nvoid func() { b::bar b; }\n} "
            "// namespace a",
            runIncludeFixer("namespace a {\nvoid func() { b::bar b; }\n}"));
  EXPECT_EQ("namespace A { c::b::bar b; }\n",
            runIncludeFixer("namespace A { c::b::bar b; }\n"));
  // FIXME: The header should not be added here. Remove this after we support
  // full match.
  EXPECT_EQ("#include \"bar.h\"\nnamespace A {\na::b::bar b;\n}",
            runIncludeFixer("namespace A {\nb::bar b;\n}"));

  // Finds candidates for "str::StrCat".
  EXPECT_EQ("#include \"strcat.h\"\nnamespace foo2 {\nstr::StrCat b;\n}",
            runIncludeFixer("namespace foo2 {\nstr::StrCat b;\n}"));
  // str::StrCat2 doesn't exist.
  // In these two cases, StrCat2 is a nested class of class str.
  EXPECT_EQ("#include \"str.h\"\nnamespace foo2 {\nstr::StrCat2 b;\n}",
            runIncludeFixer("namespace foo2 {\nstr::StrCat2 b;\n}"));
  EXPECT_EQ("#include \"str.h\"\nnamespace ns {\nstr::StrCat2 b;\n}",
            runIncludeFixer("namespace ns {\nstr::StrCat2 b;\n}"));
}

TEST(IncludeFixer, EnumConstantSymbols) {
  EXPECT_EQ("#include \"color.h\"\nint test = a::b::Green;\n",
            runIncludeFixer("int test = a::b::Green;\n"));
}

TEST(IncludeFixer, IgnoreSymbolFromHeader) {
  std::string Code = "#include \"header.h\"";
  EXPECT_EQ(Code, runIncludeFixer(Code));
}

// FIXME: add test cases for inserting and sorting multiple headers when
// include-fixer supports multiple headers insertion.
TEST(IncludeFixer, InsertAndSortSingleHeader) {
  // Insert one header.
  std::string Code = "#include \"a.h\"\n"
                     "#include \"foo.h\"\n"
                     "\n"
                     "namespace a {\nb::bar b;\n}\n";
  std::string Expected = "#include \"a.h\"\n"
                         "#include \"bar.h\"\n"
                         "#include \"foo.h\"\n"
                         "\n"
                         "namespace a {\nb::bar b;\n}\n";
  EXPECT_EQ(Expected, runIncludeFixer(Code));
}

TEST(IncludeFixer, DoNotDeleteMatchedSymbol) {
  EXPECT_EQ("#include \"Vector.h\"\na::Vector v;",
            runIncludeFixer("a::Vector v;"));
}

TEST(IncludeFixer, FixNamespaceQualifiers) {
  EXPECT_EQ("#include \"bar.h\"\na::b::bar b;\n",
            runIncludeFixer("b::bar b;\n"));
  EXPECT_EQ("#include \"bar.h\"\na::b::bar b;\n",
            runIncludeFixer("a::b::bar b;\n"));
  EXPECT_EQ("#include \"bar.h\"\na::b::bar b;\n",
            runIncludeFixer("bar b;\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nb::bar b;\n}\n",
            runIncludeFixer("namespace a {\nb::bar b;\n}\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nb::bar b;\n}\n",
            runIncludeFixer("namespace a {\nbar b;\n}\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nnamespace b{\nbar b;\n}\n} "
            "// namespace a\n",
            runIncludeFixer("namespace a {\nnamespace b{\nbar b;\n}\n}\n"));
  EXPECT_EQ("c::b::bar b;\n",
            runIncludeFixer("c::b::bar b;\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace d {\na::b::bar b;\n}\n",
            runIncludeFixer("namespace d {\nbar b;\n}\n"));
  EXPECT_EQ("#include \"bar2.h\"\nnamespace c {\na::c::bar b;\n}\n",
            runIncludeFixer("namespace c {\nbar b;\n}\n"));

  // Test common qualifers reduction.
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nnamespace d {\nb::bar b;\n}\n} "
            "// namespace a\n",
            runIncludeFixer("namespace a {\nnamespace d {\nbar b;\n}\n}\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace d {\nnamespace a {\na::b::bar "
            "b;\n}\n} // namespace d\n",
            runIncludeFixer("namespace d {\nnamespace a {\nbar b;\n}\n}\n"));

  // Test nested classes.
  EXPECT_EQ("#include \"bar.h\"\nnamespace d {\na::b::bar::t b;\n}\n",
            runIncludeFixer("namespace d {\nbar::t b;\n}\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace c {\na::b::bar::t b;\n}\n",
            runIncludeFixer("namespace c {\nbar::t b;\n}\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\nb::bar::t b;\n}\n",
            runIncludeFixer("namespace a {\nbar::t b;\n}\n"));

  EXPECT_EQ("#include \"color.h\"\nint test = a::b::Green;\n",
            runIncludeFixer("int test = Green;\n"));
  EXPECT_EQ("#include \"color.h\"\nnamespace d {\nint test = a::b::Green;\n}\n",
            runIncludeFixer("namespace d {\nint test = Green;\n}\n"));
  EXPECT_EQ("#include \"color.h\"\nnamespace a {\nint test = b::Green;\n}\n",
            runIncludeFixer("namespace a {\nint test = Green;\n}\n"));

  // Test global scope operator.
  EXPECT_EQ("#include \"bar.h\"\n::a::b::bar b;\n",
            runIncludeFixer("::a::b::bar b;\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a {\n::a::b::bar b;\n}\n",
            runIncludeFixer("namespace a {\n::a::b::bar b;\n}\n"));
}

TEST(IncludeFixer, FixNamespaceQualifiersForAllInstances) {
  const char TestCode[] = R"(
namespace a {
bar b;
int func1() {
  bar a;
                                                             bar *p = new bar();
  return 0;
}
} // namespace a

namespace a {
bar func2() {
  bar f;
  return f;
}
} // namespace a

// Non-fixed cases:
void f() {
  bar b;
}

namespace a {
namespace c {
  bar b;
} // namespace c
} // namespace a
)";

  const char ExpectedCode[] = R"(
#include "bar.h"
namespace a {
b::bar b;
int func1() {
  b::bar a;
  b::bar *p = new b::bar();
  return 0;
}
} // namespace a

namespace a {
b::bar func2() {
  b::bar f;
  return f;
}
} // namespace a

// Non-fixed cases:
void f() {
  bar b;
}

namespace a {
namespace c {
  bar b;
} // namespace c
} // namespace a
)";

  EXPECT_EQ(ExpectedCode, runIncludeFixer(TestCode));
}

TEST(IncludeFixer, DontAddQualifiersForMissingCompleteType) {
  EXPECT_EQ("#include \"bar.h\"\nclass bar;\nvoid f() {\nbar* b;\nb->f();\n}",
            runIncludeFixer("class bar;\nvoid f() {\nbar* b;\nb->f();\n}"));
}

} // namespace
} // namespace include_fixer
} // namespace clang
