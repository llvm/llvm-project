//===- ExpandResponseFileCompilationDataBase.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace tooling {
namespace {

class ExpandResponseFilesDatabase : public CompilationDatabase {
public:
  ExpandResponseFilesDatabase(
      std::unique_ptr<CompilationDatabase> Base,
      llvm::cl::TokenizerCallback Tokenizer,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
      : Base(std::move(Base)), Tokenizer(Tokenizer), FS(std::move(FS)) {
    assert(this->Base != nullptr);
    assert(this->Tokenizer != nullptr);
    assert(this->FS != nullptr);
  }

  std::vector<std::string> getAllFiles() const override {
    return Base->getAllFiles();
  }

  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return expand(Base->getCompileCommands(FilePath));
  }

  std::vector<CompileCommand> getAllCompileCommands() const override {
    return expand(Base->getAllCompileCommands());
  }

private:
  std::vector<CompileCommand> expand(std::vector<CompileCommand> Cmds) const {
    for (auto &Cmd : Cmds) {
      bool SeenRSPFile = false;
      llvm::SmallVector<const char *, 20> Argv;
      Argv.reserve(Cmd.CommandLine.size());
      for (auto &Arg : Cmd.CommandLine) {
        Argv.push_back(Arg.c_str());
        if (!Arg.empty())
          SeenRSPFile |= Arg.front() == '@';
      }
      if (!SeenRSPFile)
        continue;
      llvm::BumpPtrAllocator Alloc;
      llvm::StringSaver Saver(Alloc);
      llvm::cl::ExpandResponseFiles(Saver, Tokenizer, Argv, false, false, false,
                                    llvm::StringRef(Cmd.Directory), *FS);
      // Don't assign directly, Argv aliases CommandLine.
      std::vector<std::string> ExpandedArgv(Argv.begin(), Argv.end());
      Cmd.CommandLine = std::move(ExpandedArgv);
    }
    return Cmds;
  }

private:
  std::unique_ptr<CompilationDatabase> Base;
  llvm::cl::TokenizerCallback Tokenizer;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
};

} // namespace

std::unique_ptr<CompilationDatabase>
expandResponseFiles(std::unique_ptr<CompilationDatabase> Base,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  auto Tokenizer = llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()
                       ? llvm::cl::TokenizeWindowsCommandLine
                       : llvm::cl::TokenizeGNUCommandLine;
  return std::make_unique<ExpandResponseFilesDatabase>(
      std::move(Base), Tokenizer, std::move(FS));
}

} // namespace tooling
} // namespace clang
