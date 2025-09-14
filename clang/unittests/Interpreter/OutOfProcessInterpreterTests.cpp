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
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
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

static std::string getOrcRuntimePath() {
  llvm::SmallString<256> RuntimePath(llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getOrcRuntimePath)));
  removePathComponent(5, RuntimePath);
  llvm::sys::path::append(RuntimePath, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                          CLANG_VERSION_MAJOR_STRING, "lib");

  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());
  if (SystemTriple.isOSBinFormatMachO()) {
    llvm::sys::path::append(RuntimePath, "darwin", "liborc_rt_osx.a");
  } else if (SystemTriple.isOSBinFormatELF()) {
    llvm::sys::path::append(RuntimePath, "x86_64-unknown-linux-gnu",
                            "liborc_rt.a");
  }
  return RuntimePath.str().str();
}

static std::unique_ptr<Interpreter>
createInterpreterWithRemoteExecution(std::shared_ptr<IOContext> io_ctx,
                                     const Args &ExtraArgs = {}) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  llvm::append_range(ClangArgs, ExtraArgs);
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgs);
  auto CI = cantFail(CB.CreateCpp());

  clang::Interpreter::JITConfig Config;
  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());

  if (SystemTriple.isOSBinFormatELF() || SystemTriple.isOSBinFormatMachO()) {
    Config.IsOutOfProcess = true;
    Config.OOPExecutor = getExecutorPath();
    Config.UseSharedMemory = false;
    Config.SlabAllocateSize = 0;
    Config.OrcRuntimePath = getOrcRuntimePath();

    int stdin_fd = fileno(io_ctx->stdin_file.get());
    int stdout_fd = fileno(io_ctx->stdout_file.get());
    int stderr_fd = fileno(io_ctx->stderr_file.get());

    Config.CustomizeFork = [=] {
      auto redirect = [](int from, int to) {
        if (from != to) {
          dup2(from, to);
          close(from);
        }
      };

      redirect(stdin_fd, STDIN_FILENO);
      redirect(stdout_fd, STDOUT_FILENO);
      redirect(stderr_fd, STDERR_FILENO);

      setvbuf(stdout, nullptr, _IONBF, 0);
      setvbuf(stderr, nullptr, _IONBF, 0);

      printf("CustomizeFork executed\n");
      fflush(stdout);
    };
  }

  return cantFail(clang::Interpreter::create(std::move(CI), Config));
}

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST_F(InterpreterTestBase, SanityWithRemoteExecution) {
  if (!HostSupportsJIT())
    GTEST_SKIP();

  std::string OrcRuntimePath = getOrcRuntimePath();
  std::string ExecutorPath = getExecutorPath();

  if (!llvm::sys::fs::exists(OrcRuntimePath) ||
      !llvm::sys::fs::exists(ExecutorPath))
    GTEST_SKIP();

  auto io_ctx = std::make_shared<IOContext>();
  ASSERT_TRUE(io_ctx->initializeTempFiles());

  std::unique_ptr<Interpreter> Interp =
      createInterpreterWithRemoteExecution(io_ctx);
  ASSERT_TRUE(Interp);

  using PTU = PartialTranslationUnit;
  PTU &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, DeclsSize(R1.TUPart));

  PTU &R2(cantFail(Interp->Parse("int i = 42;")));
  EXPECT_EQ(1U, DeclsSize(R2.TUPart));

  std::string captured_stdout = io_ctx->readStdoutContent();
  std::string captured_stderr = io_ctx->readStderrContent();

  EXPECT_TRUE(captured_stdout.find("CustomizeFork executed") !=
              std::string::npos);
}

} // end anonymous namespace