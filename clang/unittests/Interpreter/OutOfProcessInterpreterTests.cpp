//===- unittests/Interpreter/OutOfProcessInterpreterTest.cpp --- Interpreter
// tests when Out-of-Process ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Clang's Interpreter library.
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Interpreter/IncrementalExecutor.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <signal.h>
#include <sstream>
#include <unistd.h>

using namespace clang;

llvm::ExitOnError ExitOnError;

namespace {

using Args = std::vector<const char *>;

struct FileDeleter {
  void operator()(FILE *f) {
    if (f)
      fclose(f);
  }
};

struct IOContext {
  std::unique_ptr<FILE, FileDeleter> stdin_file;
  std::unique_ptr<FILE, FileDeleter> stdout_file;
  std::unique_ptr<FILE, FileDeleter> stderr_file;

  bool initializeTempFiles() {
    stdin_file.reset(tmpfile());
    stdout_file.reset(tmpfile());
    stderr_file.reset(tmpfile());
    return stdin_file && stdout_file && stderr_file;
  }

  std::string readStdoutContent() {
    if (!stdout_file)
      return "";
    rewind(stdout_file.get());
    std::ostringstream content;
    char buffer[1024];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), stdout_file.get())) >
           0) {
      content.write(buffer, bytes_read);
    }
    return content.str();
  }

  std::string readStderrContent() {
    if (!stderr_file)
      return "";
    rewind(stderr_file.get());
    std::ostringstream content;
    char buffer[1024];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), stderr_file.get())) >
           0) {
      content.write(buffer, bytes_read);
    }
    return content.str();
  }
};

static void removePathComponent(unsigned N, llvm::SmallString<256> &Path) {
  for (unsigned i = 0; i < N; ++i)
    llvm::sys::path::remove_filename(Path);
}

static std::string getExecutorPath() {
  llvm::SmallString<256> ExecutorPath(llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getExecutorPath)));
  removePathComponent(5, ExecutorPath);
  llvm::sys::path::append(ExecutorPath, "bin", "llvm-jitlink-executor");
  return ExecutorPath.str().str();
}

struct OutOfProcessInterpreterInfo {
  std::string OrcRuntimePath;
  std::unique_ptr<Interpreter> Interp;
};

static llvm::Expected<OutOfProcessInterpreterInfo>
createInterpreterWithRemoteExecution(std::shared_ptr<IOContext> io_ctx,
                                     const Args &ExtraArgs = {}) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  llvm::append_range(ClangArgs, ExtraArgs);

  auto Config = std::make_unique<IncrementalExecutorBuilder>();

  Config->IsOutOfProcess = true;
  Config->OOPExecutor = getExecutorPath();
  Config->UseSharedMemory = false;
  Config->SlabAllocateSize = 0;

  // Capture the raw file descriptors by value explicitly. This lambda will
  // be invoked in the child process after fork(), so capturing the fd ints is
  // safe and avoids capturing FILE* pointers or outer 'this'.
  int stdin_fd = fileno(io_ctx->stdin_file.get());
  int stdout_fd = fileno(io_ctx->stdout_file.get());
  int stderr_fd = fileno(io_ctx->stderr_file.get());

  Config->CustomizeFork = [stdin_fd, stdout_fd, stderr_fd]() {
    auto redirect = [](int from, int to) {
      if (from != to) {
        dup2(from, to);
        close(from);
      }
    };

    redirect(stdin_fd, STDIN_FILENO);
    redirect(stdout_fd, STDOUT_FILENO);
    redirect(stderr_fd, STDERR_FILENO);

    // Unbuffer the stdio in the child; useful for deterministic tests.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Helpful marker for the unit-test to assert that fork customization ran.
    printf("CustomizeFork executed\n");
    fflush(stdout);
  };
  auto CB = IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgs);
  CB.SetDriverCompilationCallback(Config->UpdateOrcRuntimePathCB);

  auto CIOrErr = CB.CreateCpp();
  if (!CIOrErr)
    return CIOrErr.takeError();

  OutOfProcessInterpreterInfo Info;
  Info.OrcRuntimePath = Config->OrcRuntimePath;

  auto InterpOrErr =
      Interpreter::create(std::move(*CIOrErr), std::move(Config));
  if (!InterpOrErr)
    return InterpOrErr.takeError();

  Info.Interp = std::move(*InterpOrErr);
  return Info;
}

class OutOfProcessInterpreterTest : public InterpreterTestBase {
protected:
  static bool HostSupportsOutOfProcessJIT() {
    if (!InterpreterTestBase::HostSupportsJIT())
      return false;

    llvm::Triple SystemTriple(llvm::sys::getProcessTriple());
    if (!SystemTriple.isOSBinFormatELF() && !SystemTriple.isOSBinFormatMachO())
      return false;

    return !getExecutorPath().empty();
  }

  static void SetUpTestSuite() {
    if (!HostSupportsOutOfProcessJIT())
      GTEST_SKIP() << "Host does not support out-of-process JIT";

    auto io_ctx = std::make_shared<IOContext>();
    if (!io_ctx->initializeTempFiles())
      GTEST_SKIP() << "Cannot initialize temporary files";

    auto ErrOrI = createInterpreterWithRemoteExecution(io_ctx);
    if (!ErrOrI) {
      consumeError(llvm::handleErrors(
          ErrOrI.takeError(), [](const llvm::StringError &SE) {
            // check for your specific error code
            if (SE.convertToErrorCode() ==
                std::errc::no_such_file_or_directory) {
              // skip the test
              GTEST_SKIP() << "File not found: " << SE.getMessage();
            }
          }));
    }
  }
};

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST_F(OutOfProcessInterpreterTest, SanityWithRemoteExecution) {
  auto io_ctx = std::make_shared<IOContext>();
  ASSERT_TRUE(io_ctx->initializeTempFiles());

  OutOfProcessInterpreterInfo Info =
      cantFail(createInterpreterWithRemoteExecution(io_ctx));
  Interpreter *Interp = Info.Interp.get();
  ASSERT_TRUE(Interp);

  using PTU = PartialTranslationUnit;
  PTU &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, DeclsSize(R1.TUPart));

  PTU &R2(cantFail(Interp->Parse("int i = 42;")));
  EXPECT_EQ(1U, DeclsSize(R2.TUPart));

  std::string captured_stdout = io_ctx->readStdoutContent();
  std::string captured_stderr = io_ctx->readStderrContent();

  EXPECT_NE(std::string::npos, captured_stdout.find("CustomizeFork executed"));
}

TEST_F(OutOfProcessInterpreterTest, FindRuntimeInterface) {
  // make a fresh io context for this test
  auto io_ctx = std::make_shared<IOContext>();
  ASSERT_TRUE(io_ctx->initializeTempFiles());

  OutOfProcessInterpreterInfo I =
      cantFail(createInterpreterWithRemoteExecution(io_ctx));
  ASSERT_TRUE(I.Interp);

  // FIXME: Not yet supported.
  // cantFail(I->Parse("int a = 1; a"));
  // cantFail(I->Parse("int b = 2; b"));
  // cantFail(I->Parse("int c = 3; c"));

  // // Make sure no clang::Value logic is attached by the Interpreter.
  // Value V1;
  // llvm::cantFail(I->ParseAndExecute("int x = 42;"));
  // llvm::cantFail(I->ParseAndExecute("x", &V1));
  // EXPECT_FALSE(V1.isValid());
  // EXPECT_FALSE(V1.hasValue());
}

} // end anonymous namespace
