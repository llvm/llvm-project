//===--- RewriterTestContext.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a utility class for Rewriter related tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_TOOLING_REWRITERTESTCONTEXT_H
#define LLVM_CLANG_UNITTESTS_TOOLING_REWRITERTESTCONTEXT_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

/// \brief A very simple diagnostic consumer that prints to stderr and keeps
/// track of the number of diagnostics.
///
/// This avoids a dependency on clangFrontend for FormatTests.
struct RewriterDiagnosticConsumer : public DiagnosticConsumer {
  RewriterDiagnosticConsumer() : NumDiagnosticsSeen(0) {}
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    ++NumDiagnosticsSeen;
    SmallString<100> OutStr;
    Info.FormatDiagnostic(OutStr);
    llvm::errs() << OutStr;
  }
  unsigned NumDiagnosticsSeen;
};

/// \brief A class that sets up a ready to use Rewriter.
///
/// Useful in unit tests that need a Rewriter. Creates all dependencies
/// of a Rewriter with default values for testing and provides convenience
/// methods, which help with writing tests that change files.
class RewriterTestContext {
 public:
   RewriterTestContext()
       : Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                     DiagOpts),
         InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
         OverlayFileSystem(
             new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem())),
         Files(FileSystemOptions(), OverlayFileSystem),
         Sources(Diagnostics, Files), Rewrite(Sources, Options) {
     Diagnostics.setClient(&DiagnosticPrinter, false);
     // FIXME: To make these tests truly in-memory, we need to overlay the
     // builtin headers.
     OverlayFileSystem->pushOverlay(InMemoryFileSystem);
   }

  ~RewriterTestContext() {}

  FileID createInMemoryFile(StringRef Name, StringRef Content) {
    std::unique_ptr<llvm::MemoryBuffer> Source =
        llvm::MemoryBuffer::getMemBuffer(Content);
    InMemoryFileSystem->addFile(Name, 0, std::move(Source));

    auto Entry = Files.getOptionalFileRef(Name);
    assert(Entry);
    return Sources.createFileID(*Entry, SourceLocation(), SrcMgr::C_User);
  }

  // FIXME: this code is mostly a duplicate of
  // unittests/Tooling/RefactoringTest.cpp. Figure out a way to share it.
  FileID createOnDiskFile(StringRef Name, StringRef Content) {
    SmallString<1024> Path;
    int FD;
    std::error_code EC = llvm::sys::fs::createTemporaryFile(Name, "", FD, Path);
    assert(!EC);
    (void)EC;

    llvm::raw_fd_ostream OutStream(FD, true);
    OutStream << Content;
    OutStream.close();
    auto File = Files.getOptionalFileRef(Path);
    assert(File);

    StringRef Found =
        TemporaryFiles.insert(std::make_pair(Name, std::string(Path.str())))
            .first->second;
    assert(Found == Path);
    (void)Found;
    return Sources.createFileID(*File, SourceLocation(), SrcMgr::C_User);
  }

  SourceLocation getLocation(FileID ID, unsigned Line, unsigned Column) {
    SourceLocation Result = Sources.translateFileLineCol(
        Sources.getFileEntryForID(ID), Line, Column);
    assert(Result.isValid());
    return Result;
  }

  std::string getRewrittenText(FileID ID) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Rewrite.getEditBuffer(ID).write(OS);
    return Result;
  }

  std::string getFileContentFromDisk(StringRef Name) {
    std::string Path = TemporaryFiles.lookup(Name);
    assert(!Path.empty());
    // We need to read directly from the FileManager without relaying through
    // a FileEntry, as otherwise we'd read through an already opened file
    // descriptor, which might not see the changes made.
    // FIXME: Figure out whether there is a way to get the SourceManger to
    // reopen the file.
    auto FileBuffer = Files.getBufferForFile(Path);
    return std::string((*FileBuffer)->getBuffer());
  }

  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diagnostics;
  RewriterDiagnosticConsumer DiagnosticPrinter;
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem;
  FileManager Files;
  SourceManager Sources;
  LangOptions Options;
  Rewriter Rewrite;

  // Will be set once on disk files are generated.
  llvm::StringMap<std::string> TemporaryFiles;
};

} // end namespace clang

#endif
