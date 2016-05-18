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

static bool runOnCode(tooling::ToolAction *ToolAction, StringRef Code,
                      StringRef FileName,
                      const std::vector<std::string> &ExtraArgs) {
  llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), InMemoryFileSystem));
  std::vector<std::string> Args = {"include_fixer", "-fsyntax-only", FileName};
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
  return Invocation.run();
}

static std::string runIncludeFixer(
    StringRef Code,
    const std::vector<std::string> &ExtraArgs = std::vector<std::string>()) {
  std::vector<SymbolInfo> Symbols = {
      SymbolInfo("string", SymbolInfo::SymbolKind::Class, "<string>", 1,
                 {{SymbolInfo::ContextType::Namespace, "std"}}),
      SymbolInfo("sting", SymbolInfo::SymbolKind::Class, "\"sting\"", 1,
                 {{SymbolInfo::ContextType::Namespace, "std"}}),
      SymbolInfo("size_type", SymbolInfo::SymbolKind::Variable, "<string>", 1,
                 {{SymbolInfo::ContextType::Namespace, "string"},
                  {SymbolInfo::ContextType::Namespace, "std"}}),
      SymbolInfo("foo", SymbolInfo::SymbolKind::Class, "\"dir/otherdir/qux.h\"",
                 1, {{SymbolInfo::ContextType::Namespace, "b"},
                     {SymbolInfo::ContextType::Namespace, "a"}}),
      SymbolInfo("bar", SymbolInfo::SymbolKind::Class, "\"bar.h\"",
                 1, {{SymbolInfo::ContextType::Namespace, "b"},
                     {SymbolInfo::ContextType::Namespace, "a"}}),
      SymbolInfo("Green", SymbolInfo::SymbolKind::Class, "\"color.h\"",
                 1, {{SymbolInfo::ContextType::EnumDecl, "Color"},
                     {SymbolInfo::ContextType::Namespace, "b"},
                     {SymbolInfo::ContextType::Namespace, "a"}}),
  };
  auto SymbolIndexMgr = llvm::make_unique<include_fixer::SymbolIndexManager>();
  SymbolIndexMgr->addSymbolIndex(
      llvm::make_unique<include_fixer::InMemorySymbolIndex>(Symbols));

  std::vector<clang::tooling::Replacement> Replacements;
  IncludeFixerActionFactory Factory(*SymbolIndexMgr, Replacements);
  runOnCode(&Factory, Code, "input.cc", ExtraArgs);
  clang::RewriterTestContext Context;
  clang::FileID ID = Context.createInMemoryFile("input.cc", Code);
  clang::tooling::applyAllReplacements(Replacements, Context.Rewrite);
  return Context.getRewrittenText(ID);
}

TEST(IncludeFixer, Typo) {
  EXPECT_EQ("#include <string>\nstd::string foo;\n",
            runIncludeFixer("std::string foo;\n"));

  EXPECT_EQ(
      "// comment\n#include <string>\n#include \"foo.h\"\nstd::string foo;\n"
      "#include \"dir/bar.h\"\n",
      runIncludeFixer("// comment\n#include \"foo.h\"\nstd::string foo;\n"
                      "#include \"dir/bar.h\"\n"));

  EXPECT_EQ("#include <string>\n#include \"foo.h\"\nstd::string foo;\n",
            runIncludeFixer("#include \"foo.h\"\nstd::string foo;\n"));

  EXPECT_EQ(
      "#include <string>\n#include \"foo.h\"\nstd::string::size_type foo;\n",
      runIncludeFixer("#include \"foo.h\"\nstd::string::size_type foo;\n"));

  // string without "std::" can also be fixed since fixed db results go through
  // SymbolIndexManager, and SymbolIndexManager matches unqualified identifiers
  // too.
  EXPECT_EQ("#include <string>\nstring foo;\n",
            runIncludeFixer("string foo;\n"));
}

TEST(IncludeFixer, IncompleteType) {
  EXPECT_EQ(
      "#include <string>\n#include \"foo.h\"\n"
      "namespace std {\nclass string;\n}\nstring foo;\n",
      runIncludeFixer("#include \"foo.h\"\n"
                      "namespace std {\nclass string;\n}\nstring foo;\n"));
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

#ifndef _WIN32
// It doesn't pass for targeting win32. Investigating.
TEST(IncludeFixer, NestedName) {
  // Some tests don't pass for target *-win32.
  std::vector<std::string> args = {"-target", "x86_64-unknown-unknown"};
  EXPECT_EQ("#include \"dir/otherdir/qux.h\"\n"
            "int x = a::b::foo(0);\n",
            runIncludeFixer("int x = a::b::foo(0);\n", args));

  // FIXME: Handle simple macros.
  EXPECT_EQ("#define FOO a::b::foo\nint x = FOO;\n",
            runIncludeFixer("#define FOO a::b::foo\nint x = FOO;\n"));
  EXPECT_EQ("#define FOO(x) a::##x\nint x = FOO(b::foo);\n",
            runIncludeFixer("#define FOO(x) a::##x\nint x = FOO(b::foo);\n"));

  EXPECT_EQ("#include \"dir/otherdir/qux.h\"\n"
            "namespace a {}\nint a = a::b::foo(0);\n",
            runIncludeFixer("namespace a {}\nint a = a::b::foo(0);\n", args));
}
#endif

TEST(IncludeFixer, MultipleMissingSymbols) {
  EXPECT_EQ("#include <string>\nstd::string bar;\nstd::sting foo;\n",
            runIncludeFixer("std::string bar;\nstd::sting foo;\n"));
}

TEST(IncludeFixer, ScopedNamespaceSymbols) {
  EXPECT_EQ("#include \"bar.h\"\nnamespace a { b::bar b; }\n",
            runIncludeFixer("namespace a { b::bar b; }\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace A { a::b::bar b; }\n",
            runIncludeFixer("namespace A { a::b::bar b; }\n"));
  EXPECT_EQ("#include \"bar.h\"\nnamespace a { void func() { b::bar b; } }\n",
            runIncludeFixer("namespace a { void func() { b::bar b; } }\n"));
  EXPECT_EQ("namespace A { c::b::bar b; }\n",
            runIncludeFixer("namespace A { c::b::bar b; }\n"));
  // FIXME: The header should not be added here. Remove this after we support
  // full match.
  EXPECT_EQ("#include \"bar.h\"\nnamespace A { b::bar b; }\n",
            runIncludeFixer("namespace A { b::bar b; }\n"));
}

TEST(IncludeFixer, EnumConstantSymbols) {
  EXPECT_EQ("#include \"color.h\"\nint test = a::b::Green;\n",
            runIncludeFixer("int test = a::b::Green;\n"));
}

} // namespace
} // namespace include_fixer
} // namespace clang
