//===----- EditedSource.h - Collection of source edits ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EDIT_EDITEDSOURCE_H
#define LLVM_CLANG_EDIT_EDITEDSOURCE_H

#include "clang/Edit/FileOffset.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace clang {
  class LangOptions;
  class PreprocessingRecord;

namespace edit {
  class Commit;
  class EditsReceiver;

class EditedSource {
  const SourceManager &SourceMgr;
  const LangOptions &LangOpts;
  const PreprocessingRecord *PPRec;

  struct FileEdit {
    StringRef Text;
    unsigned RemoveLen;

    FileEdit() : RemoveLen(0) {}
  };

  typedef std::map<FileOffset, FileEdit> FileEditsTy;
  FileEditsTy FileEdits;

  llvm::DenseMap<unsigned, SourceLocation> ExpansionToArgMap;

  llvm::BumpPtrAllocator StrAlloc;

public:
  EditedSource(const SourceManager &SM, const LangOptions &LangOpts,
               const PreprocessingRecord *PPRec = 0)
    : SourceMgr(SM), LangOpts(LangOpts), PPRec(PPRec),
      StrAlloc(/*size=*/512) { }

  const SourceManager &getSourceManager() const { return SourceMgr; }
  const LangOptions &getLangOpts() const { return LangOpts; }
  const PreprocessingRecord *getPreprocessingRecord() const { return PPRec; }

  bool canInsertInOffset(SourceLocation OrigLoc, FileOffset Offs);

  bool commit(const Commit &commit);
  
  void applyRewrites(EditsReceiver &receiver);
  void clearRewrites();

  StringRef copyString(StringRef str) {
    char *buf = StrAlloc.Allocate<char>(str.size());
    std::memcpy(buf, str.data(), str.size());
    return StringRef(buf, str.size());
  }
  StringRef copyString(const Twine &twine);

private:
  bool commitInsert(SourceLocation OrigLoc, FileOffset Offs, StringRef text,
                    bool beforePreviousInsertions);
  bool commitInsertFromRange(SourceLocation OrigLoc, FileOffset Offs,
                             FileOffset InsertFromRangeOffs, unsigned Len,
                             bool beforePreviousInsertions);
  void commitRemove(SourceLocation OrigLoc, FileOffset BeginOffs, unsigned Len);

  StringRef getSourceText(FileOffset BeginOffs, FileOffset EndOffs,
                          bool &Invalid);
  FileEditsTy::iterator getActionForOffset(FileOffset Offs);
};

}

} // end namespace clang

#endif
