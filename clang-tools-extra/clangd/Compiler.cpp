//===--- Compiler.cpp --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "Logger.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace clangd {

void IgnoreDiagnostics::log(DiagnosticsEngine::Level DiagLevel,
                            const clang::Diagnostic &Info) {
  // FIXME: format lazily, in case vlog is off.
  llvm::SmallString<64> Message;
  Info.FormatDiagnostic(Message);

  llvm::SmallString<64> Location;
  if (Info.hasSourceManager() && Info.getLocation().isValid()) {
    auto &SourceMgr = Info.getSourceManager();
    auto Loc = SourceMgr.getFileLoc(Info.getLocation());
    llvm::raw_svector_ostream OS(Location);
    Loc.print(OS, SourceMgr);
    OS << ":";
  }

  clangd::vlog("Ignored diagnostic. {0}{1}", Location, Message);
}

void IgnoreDiagnostics::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                         const clang::Diagnostic &Info) {
  IgnoreDiagnostics::log(DiagLevel, Info);
}

std::unique_ptr<CompilerInstance>
prepareCompilerInstance(std::unique_ptr<clang::CompilerInvocation> CI,
                        const PrecompiledPreamble *Preamble,
                        std::unique_ptr<llvm::MemoryBuffer> Buffer,
                        std::shared_ptr<PCHContainerOperations> PCHs,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                        DiagnosticConsumer &DiagsClient) {
  assert(VFS && "VFS is null");
  assert(!CI->getPreprocessorOpts().RetainRemappedFileBuffers &&
         "Setting RetainRemappedFileBuffers to true will cause a memory leak "
         "of ContentsBuffer");

  // NOTE: we use Buffer.get() when adding remapped files, so we have to make
  // sure it will be released if no error is emitted.
  if (Preamble) {
    Preamble->OverridePreamble(*CI, VFS, Buffer.get());
  } else {
    CI->getPreprocessorOpts().addRemappedFile(
        CI->getFrontendOpts().Inputs[0].getFile(), Buffer.get());
  }

  auto Clang = llvm::make_unique<CompilerInstance>(PCHs);
  Clang->setInvocation(std::move(CI));
  Clang->createDiagnostics(&DiagsClient, false);

  if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics(), VFS))
    VFS = VFSWithRemapping;
  Clang->setVirtualFileSystem(VFS);

  Clang->setTarget(TargetInfo::CreateTargetInfo(
      Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
  if (!Clang->hasTarget())
    return nullptr;

  // RemappedFileBuffers will handle the lifetime of the Buffer pointer,
  // release it.
  Buffer.release();
  return Clang;
}

} // namespace clangd
} // namespace clang
