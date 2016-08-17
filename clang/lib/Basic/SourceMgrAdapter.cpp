//=== SourceMgrAdapter.cpp - SourceMgr to SourceManager Adapter -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the adapter that maps diagnostics from llvm::SourceMgr
// to Clang's SourceManager.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceMgrAdapter.h"
#include "clang/Basic/Diagnostic.h"

using namespace clang;

void SourceMgrAdapter::handleDiag(const llvm::SMDiagnostic &diag,
                                  void *context) {
  static_cast<SourceMgrAdapter *>(context)->handleDiag(diag);
}

SourceMgrAdapter::SourceMgrAdapter(SourceManager &srcMgr,
                                   DiagnosticsEngine &diag,
                                   unsigned errorDiagID,
                                   unsigned warningDiagID,
                                   unsigned noteDiagID,
                                   const FileEntry *defaultFile)
  : SrcMgr(srcMgr), Diag(diag), ErrorDiagID(errorDiagID),
    WarningDiagID(warningDiagID), NoteDiagID(noteDiagID),
    DefaultFile(defaultFile) { }

SourceMgrAdapter::~SourceMgrAdapter() { }

SourceLocation SourceMgrAdapter::mapLocation(const llvm::SourceMgr &llvmSrcMgr,
                                             llvm::SMLoc loc) {
  // Map invalid locations.
  if (!loc.isValid())
    return SourceLocation();

  // Find the buffer containing the location.
  unsigned bufferID = llvmSrcMgr.FindBufferContainingLoc(loc);
  if (!bufferID)
    return SourceLocation();


  // If we haven't seen this buffer before, copy it over.
  auto buffer = llvmSrcMgr.getMemoryBuffer(bufferID);
  auto knownBuffer = FileIDMapping.find(std::make_pair(&llvmSrcMgr, bufferID));
  if (knownBuffer == FileIDMapping.end()) {
    FileID fileID;
    if (DefaultFile) {
      // Map to the default file.
      fileID = SrcMgr.createFileID(DefaultFile, SourceLocation(),
                                   SrcMgr::C_User);

      // Only do this once.
      DefaultFile = nullptr;
    } else {
      // Make a copy of the memory buffer.
      StringRef bufferName = buffer->getBufferIdentifier();
      auto bufferCopy
        = std::unique_ptr<llvm::MemoryBuffer>(
            llvm::MemoryBuffer::getMemBufferCopy(buffer->getBuffer(),
                                                 bufferName));

      // Add this memory buffer to the Clang source manager.
      fileID = SrcMgr.createFileID(std::move(bufferCopy));
    }

    // Save the mapping.
    knownBuffer = FileIDMapping.insert(
                    std::make_pair(std::make_pair(&llvmSrcMgr, bufferID),
                                   fileID)).first;
  }

  // Translate the offset into the file.
  unsigned offset = loc.getPointer() - buffer->getBufferStart();
  return SrcMgr.getLocForStartOfFile(knownBuffer->second)
           .getLocWithOffset(offset);
}

SourceRange SourceMgrAdapter::mapRange(const llvm::SourceMgr &llvmSrcMgr,
                                       llvm::SMRange range) {
  if (!range.isValid())
    return SourceRange();

  SourceLocation start = mapLocation(llvmSrcMgr, range.Start);
  SourceLocation end = mapLocation(llvmSrcMgr, range.End);
  return SourceRange(start, end);
}

void SourceMgrAdapter::handleDiag(const llvm::SMDiagnostic &diag) {
  // Map the location.
  SourceLocation loc;
  if (auto *llvmSrcMgr = diag.getSourceMgr())
    loc = mapLocation(*llvmSrcMgr, diag.getLoc());

  // Extract the message.
  StringRef message = diag.getMessage();

  // Map the diagnostic kind.
  unsigned diagID;
  switch (diag.getKind()) {
  case llvm::SourceMgr::DK_Error:
    diagID = ErrorDiagID;
    break;

  case llvm::SourceMgr::DK_Warning:
    diagID = WarningDiagID;
    break;

  case llvm::SourceMgr::DK_Note:
    diagID = NoteDiagID;
    break;
  }

  // Report the diagnostic.
  DiagnosticBuilder builder = Diag.Report(loc, diagID) << message;

  if (auto *llvmSrcMgr = diag.getSourceMgr()) {
    // Translate ranges.
    SourceLocation startOfLine = loc.getLocWithOffset(-diag.getColumnNo());
    for (auto range : diag.getRanges()) {
      builder << SourceRange(startOfLine.getLocWithOffset(range.first),
                             startOfLine.getLocWithOffset(range.second));
    }

    // Translate Fix-Its.
    for (const llvm::SMFixIt &fixIt : diag.getFixIts()) {
      CharSourceRange range(mapRange(*llvmSrcMgr, fixIt.getRange()), false);
      builder << FixItHint::CreateReplacement(range, fixIt.getText());
    }
  }
}
