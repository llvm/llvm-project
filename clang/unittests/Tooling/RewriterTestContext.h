//===--- RewriterTestContext.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

/// \brief A class that sets up a ready to use Rewriter.
///
/// Useful in unit tests that need a Rewriter. Creates all dependencies
/// of a Rewriter with default values for testing and provides convenience
/// methods, which help with writing tests that change files.
class RewriterTestContext {
 public:
   RewriterTestContext()
       : DiagOpts(new DiagnosticOptions()),
         Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                     &*DiagOpts),
         DiagnosticPrinter(llvm::outs(), &*DiagOpts),
         InMemoryFileSystem(new vfs::InMemoryFileSystem),
         OverlayFileSystem(
             new vfs::OverlayFileSystem(vfs::getRealFileSystem())),
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

    const FileEntry *Entry = Files.getFile(Name);
    assert(Entry != nullptr);
    return Sources.createFileID(Entry, SourceLocation(), SrcMgr::C_User);
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
    const FileEntry *File = Files.getFile(Path);
    assert(File != nullptr);

    StringRef Found =
        TemporaryFiles.insert(std::make_pair(Name, Path.str())).first->second;
    assert(Found == Path);
    (void)Found;
    return Sources.createFileID(File, SourceLocation(), SrcMgr::C_User);
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
    OS.flush();
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
    return (*FileBuffer)->getBuffer();
  }

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticsEngine Diagnostics;
  TextDiagnosticPrinter DiagnosticPrinter;
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem;
  IntrusiveRefCntPtr<vfs::OverlayFileSystem> OverlayFileSystem;
  FileManager Files;
  SourceManager Sources;
  LangOptions Options;
  Rewriter Rewrite;

  // Will be set once on disk files are generated.
  llvm::StringMap<std::string> TemporaryFiles;
};

} // end namespace clang

#endif
