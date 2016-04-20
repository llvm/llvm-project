//===-- IncludeFixerTest.cpp - Include fixer unit tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "unittests/Tooling/RewriterTestContext.h"
#include "InMemoryXrefsDB.h"
#include "IncludeFixer.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
using namespace clang;

namespace clang {
namespace include_fixer {
namespace {

static bool runOnCode(tooling::ToolAction *ToolAction, StringRef Code,
                      StringRef FileName) {
  llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), InMemoryFileSystem));
  tooling::ToolInvocation Invocation(
      {std::string("include_fixer"), std::string("-fsyntax-only"),
       FileName.str()},
      ToolAction, Files.get(), std::make_shared<PCHContainerOperations>());

  InMemoryFileSystem->addFile(FileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(Code));

  InMemoryFileSystem->addFile("foo.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  InMemoryFileSystem->addFile("bar.h", 0,
                              llvm::MemoryBuffer::getMemBuffer("\n"));
  return Invocation.run();
}

static std::string runIncludeFixer(StringRef Code) {
  std::map<std::string, std::vector<std::string>> XrefsMap = {
      {"std::string", {"<string>"}}, {"std::string::size_type", {"<string>"}}};
  auto XrefsDB =
      llvm::make_unique<include_fixer::InMemoryXrefsDB>(std::move(XrefsMap));
  std::vector<clang::tooling::Replacement> Replacements;
  IncludeFixerActionFactory Factory(*XrefsDB, Replacements);
  runOnCode(&Factory, Code, "input.cc");
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
      "#include \"bar.h\"\n",
      runIncludeFixer("// comment\n#include \"foo.h\"\nstd::string foo;\n"
                      "#include \"bar.h\"\n"));

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

} // namespace
} // namespace include_fixer
} // namespace clang
