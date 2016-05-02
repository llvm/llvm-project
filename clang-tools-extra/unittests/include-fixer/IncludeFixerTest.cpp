//===-- IncludeFixerTest.cpp - Include fixer unit tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryXrefsDB.h"
#include "IncludeFixer.h"
#include "unittests/Tooling/RewriterTestContext.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
using namespace clang;

namespace clang {
namespace include_fixer {
namespace {

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
  std::map<std::string, std::vector<std::string>> XrefsMap = {
      {"std::string", {"<string>"}},
      {"std::string::size_type", {"<string>"}},
      {"a::b::foo", {"dir/otherdir/qux.h"}},
  };
  auto XrefsDB =
      llvm::make_unique<include_fixer::InMemoryXrefsDB>(std::move(XrefsMap));
  std::vector<clang::tooling::Replacement> Replacements;
  IncludeFixerActionFactory Factory(*XrefsDB, Replacements);
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

  // The fixed xrefs db doesn't know how to handle string without std::.
  EXPECT_EQ("string foo;\n", runIncludeFixer("string foo;\n"));
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

} // namespace
} // namespace include_fixer
} // namespace clang
