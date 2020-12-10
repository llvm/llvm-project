//===-- Unittests for WrapperGen ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <unistd.h>

llvm::cl::opt<std::string>
    LibcPath("path", llvm::cl::desc("Path to the top level libc directory."),
             llvm::cl::value_desc("<path to libc>"), llvm::cl::Required);
llvm::cl::opt<std::string>
    ToolPath("tool", llvm::cl::desc("Path to the tool executable."),
             llvm::cl::value_desc("<path to tool>"), llvm::cl::Required);
llvm::cl::opt<std::string>
    APIPath("api",
            llvm::cl::desc("Path to the api tablegen file used by the tests."),
            llvm::cl::value_desc("<path to testapi.td>"), llvm::cl::Required);

class WrapperGenTest : public ::testing::Test {
public:
  std::string IncludeArg;
  std::string APIArg;
  llvm::StringRef ProgPath;
  llvm::Expected<llvm::sys::fs::TempFile> STDOutFile =
      llvm::sys::fs::TempFile::create("wrappergen-stdout-%%-%%-%%-%%.txt");
  llvm::Expected<llvm::sys::fs::TempFile> STDErrFile =
      llvm::sys::fs::TempFile::create("wrappergen-stderr-%%-%%-%%-%%.txt");

protected:
  void SetUp() override {
    IncludeArg = "-I=";
    IncludeArg.append(LibcPath);
    APIArg = APIPath;
    ProgPath = llvm::StringRef(ToolPath);

    if (!STDOutFile) {
      llvm::errs() << "Error: " << llvm::toString(STDOutFile.takeError())
                   << "\n";
      llvm::report_fatal_error(
          "Temporary file failed to initialize for libc-wrappergen tests.");
    }
    if (!STDErrFile) {
      llvm::errs() << "Error: " << llvm::toString(STDErrFile.takeError())
                   << "\n";
      llvm::report_fatal_error(
          "Temporary file failed to initialize for libc-wrappergen tests.");
    }
  }
  void TearDown() override {
    llvm::consumeError(STDOutFile.get().discard());
    llvm::consumeError(STDErrFile.get().discard());
  }
};

TEST_F(WrapperGenTest, RunWrapperGenAndGetNoErrors) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef ArgV[] = {ProgPath, llvm::StringRef(IncludeArg),
                            llvm::StringRef(APIArg), "--name", "strlen"};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 0);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();
  ASSERT_EQ(STDErrOutput, "");
}

TEST_F(WrapperGenTest, RunWrapperGenOnStrlen) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef ArgV[] = {ProgPath, llvm::StringRef(IncludeArg),
                            llvm::StringRef(APIArg), "--name", "strlen"};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 0);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();

  ASSERT_EQ(STDErrOutput, "");

  auto STDOutOrError = llvm::MemoryBuffer::getFile(STDOutFile.get().TmpName);
  std::string STDOutOutput = STDOutOrError.get()->getBuffer().str();

  ASSERT_EQ(STDOutOutput, "#include \"src/string/strlen.h\"\n"
                          "extern \"C\" size_t strlen(const char * __arg0) {\n"
                          "  return __llvm_libc::strlen(__arg0);\n"
                          "}\n");
  // TODO:(michaelrj) Figure out how to make this output comparison
  // less brittle. Currently it's just comparing the output of the program
  // to an exact string, this means that even a small formatting change
  // would break this test.
}

TEST_F(WrapperGenTest, RunWrapperGenOnStrlenWithAliasee) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef ArgV[] = {ProgPath,
                            llvm::StringRef(IncludeArg),
                            llvm::StringRef(APIArg),
                            "--aliasee",
                            "STRLEN_ALIAS",
                            "--name",
                            "strlen"};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 0);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();

  ASSERT_EQ(STDErrOutput, "");

  auto STDOutOrError = llvm::MemoryBuffer::getFile(STDOutFile.get().TmpName);
  std::string STDOutOutput = STDOutOrError.get()->getBuffer().str();

  ASSERT_EQ(STDOutOutput, "extern \"C\" size_t strlen(const char * __arg0) "
                          "__attribute__((alias(\"STRLEN_ALIAS\")));\n");
  // TODO:(michaelrj) Figure out how to make this output comparison
  // less brittle. Currently it's just comparing the output of the program
  // to an exact string, this means that even a small formatting change
  // would break this test.
}

/////////////////////////////////////////////////////////////////////
// BAD INPUT TESTS
// all of the tests after this point are testing inputs that should
// return errors
/////////////////////////////////////////////////////////////////////

TEST_F(WrapperGenTest,
       RunWrapperGenOnStrlenWithAliaseeAndAliaseeFileWhichIsError) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef ArgV[] = {ProgPath,
                            llvm::StringRef(IncludeArg),
                            llvm::StringRef(APIArg),
                            "--aliasee",
                            "STRLEN_ALIAS",
                            "--aliasee-file",
                            "STRLEN_ALIAS_FILE",
                            "--name",
                            "strlen"};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 1);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();

  ASSERT_EQ(STDErrOutput, "error: The options 'aliasee' and 'aliasee-file' "
                          "cannot be specified simultaniously.\n");

  auto STDOutOrError = llvm::MemoryBuffer::getFile(STDOutFile.get().TmpName);
  std::string STDOutOutput = STDOutOrError.get()->getBuffer().str();

  ASSERT_EQ(STDOutOutput, "");
}

TEST_F(WrapperGenTest, RunWrapperGenOnBadFuncName) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef BadFuncName = "FAKE_TEST_FUNC";

  llvm::StringRef ArgV[] = {ProgPath, llvm::StringRef(IncludeArg),
                            llvm::StringRef(APIArg), "--name", BadFuncName};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 1);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();

  ASSERT_EQ(STDErrOutput, ("error: Function '" + BadFuncName +
                           "' not found in any standard spec.\n")
                              .str());

  auto STDOutOrError = llvm::MemoryBuffer::getFile(STDOutFile.get().TmpName);
  std::string STDOutOutput = STDOutOrError.get()->getBuffer().str();

  ASSERT_EQ(STDOutOutput, "");
}

TEST_F(WrapperGenTest, RunWrapperGenOnStrlenWithBadAliaseeFile) {
  llvm::Optional<llvm::StringRef> Redirects[] = {
      llvm::None, llvm::StringRef(STDOutFile.get().TmpName),
      llvm::StringRef(STDErrFile.get().TmpName)};

  llvm::StringRef BadAliaseeFileName = "FILE_THAT_DOESNT_EXIST.txt";

  llvm::StringRef ArgV[] = {
      ProgPath,         llvm::StringRef(IncludeArg), llvm::StringRef(APIArg),
      "--aliasee-file", BadAliaseeFileName,          "--name",
      "strlen"};

  int ExitCode =
      llvm::sys::ExecuteAndWait(ProgPath, ArgV, llvm::None, Redirects);

  EXPECT_EQ(ExitCode, 1);

  auto STDErrOrError = llvm::MemoryBuffer::getFile(STDErrFile.get().TmpName);
  std::string STDErrOutput = STDErrOrError.get()->getBuffer().str();

  ASSERT_EQ(STDErrOutput, ("error: Unable to read the aliasee file " +
                           BadAliaseeFileName + "\n")
                              .str());

  auto STDOutOrError = llvm::MemoryBuffer::getFile(STDOutFile.get().TmpName);
  std::string STDOutOutput = STDOutOrError.get()->getBuffer().str();

  ASSERT_EQ(STDOutOutput, "");
}
