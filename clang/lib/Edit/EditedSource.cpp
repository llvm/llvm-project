//===----- EditedSource.cpp - Collection of source edits ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Edit/EditedSource.h"
#include "clang/Edit/Commit.h"
#include "clang/Edit/EditsReceiver.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

using namespace clang;
using namespace edit;

void EditsReceiver::remove(CharSourceRange range) {
  replace(range, StringRef());
}

StringRef EditedSource::copyString(const Twine &twine) {
  llvm::SmallString<128> Data;
  return copyString(twine.toStringRef(Data));
}

bool EditedSource::canInsertInOffset(SourceLocation OrigLoc, FileOffset Offs) {
  FileEditsTy::iterator FA = getActionForOffset(Offs);
  if (FA != FileEdits.end()) {
    if (FA->first != Offs)
      return false; // position has been removed.
  }

  if (SourceMgr.isMacroArgExpansion(OrigLoc)) {
    SourceLocation
      DefArgLoc = SourceMgr.getImmediateExpansionRange(OrigLoc).first;
    SourceLocation
      ExpLoc = SourceMgr.getImmediateExpansionRange(DefArgLoc).first;
    llvm::DenseMap<unsigned, SourceLocation>::iterator
      I = ExpansionToArgMap.find(ExpLoc.getRawEncoding());
    if (I != ExpansionToArgMap.end() && I->second != DefArgLoc)
      return false; // Trying to write in a macro argument input that has
                 // already been written for another argument of the same macro. 
  }

  return true;
}

bool EditedSource::commitInsert(SourceLocation OrigLoc,
                                FileOffset Offs, StringRef text,
                                bool beforePreviousInsertions) {
  if (!canInsertInOffset(OrigLoc, Offs))
    return false;
  if (text.empty())
    return true;

  if (SourceMgr.isMacroArgExpansion(OrigLoc)) {
    SourceLocation
      DefArgLoc = SourceMgr.getImmediateExpansionRange(OrigLoc).first;
    SourceLocation
      ExpLoc = SourceMgr.getImmediateExpansionRange(DefArgLoc).first;
    ExpansionToArgMap[ExpLoc.getRawEncoding()] = DefArgLoc;
  }
  
  FileEdit &FA = FileEdits[Offs];
  if (FA.Text.empty()) {
    FA.Text = copyString(text);
    return true;
  }

  Twine concat;
  if (beforePreviousInsertions)
    concat = Twine(text) + FA.Text;
  else
    concat = Twine(FA.Text) +  text;

  FA.Text = copyString(concat);
  return true;
}

bool EditedSource::commitInsertFromRange(SourceLocation OrigLoc,
                                   FileOffset Offs,
                                   FileOffset InsertFromRangeOffs, unsigned Len,
                                   bool beforePreviousInsertions) {
  if (Len == 0)
    return true;

  llvm::SmallString<128> StrVec;
  FileOffset BeginOffs = InsertFromRangeOffs;
  FileOffset EndOffs = BeginOffs.getWithOffset(Len);
  FileEditsTy::iterator I = FileEdits.upper_bound(BeginOffs);
  if (I != FileEdits.begin())
    --I;

  for (; I != FileEdits.end(); ++I) {
    FileEdit &FA = I->second;
    FileOffset B = I->first;
    FileOffset E = B.getWithOffset(FA.RemoveLen);

    if (BeginOffs == B)
      break;

    if (BeginOffs < E) {
      if (BeginOffs > B) {
        BeginOffs = E;
        ++I;
      }
      break;
    }
  }

  for (; I != FileEdits.end() && EndOffs > I->first; ++I) {
    FileEdit &FA = I->second;
    FileOffset B = I->first;
    FileOffset E = B.getWithOffset(FA.RemoveLen);

    if (BeginOffs < B) {
      bool Invalid = false;
      StringRef text = getSourceText(BeginOffs, B, Invalid);
      if (Invalid)
        return false;
      StrVec += text;
    }
    StrVec += FA.Text;
    BeginOffs = E;
  }

  if (BeginOffs < EndOffs) {
    bool Invalid = false;
    StringRef text = getSourceText(BeginOffs, EndOffs, Invalid);
    if (Invalid)
      return false;
    StrVec += text;
  }

  return commitInsert(OrigLoc, Offs, StrVec.str(), beforePreviousInsertions);
}

void EditedSource::commitRemove(SourceLocation OrigLoc,
                                FileOffset BeginOffs, unsigned Len) {
  if (Len == 0)
    return;

  FileOffset EndOffs = BeginOffs.getWithOffset(Len);
  FileEditsTy::iterator I = FileEdits.upper_bound(BeginOffs);
  if (I != FileEdits.begin())
    --I;

  for (; I != FileEdits.end(); ++I) {
    FileEdit &FA = I->second;
    FileOffset B = I->first;
    FileOffset E = B.getWithOffset(FA.RemoveLen);

    if (BeginOffs < E)
      break;
  }

  FileOffset TopBegin, TopEnd;
  FileEdit *TopFA = 0;

  if (I == FileEdits.end()) {
    FileEditsTy::iterator
      NewI = FileEdits.insert(I, std::make_pair(BeginOffs, FileEdit()));
    NewI->second.RemoveLen = Len;
    return;
  }

  FileEdit &FA = I->second;
  FileOffset B = I->first;
  FileOffset E = B.getWithOffset(FA.RemoveLen);
  if (BeginOffs < B) {
    FileEditsTy::iterator
      NewI = FileEdits.insert(I, std::make_pair(BeginOffs, FileEdit()));
    TopBegin = BeginOffs;
    TopEnd = EndOffs;
    TopFA = &NewI->second;
    TopFA->RemoveLen = Len;
  } else {
    TopBegin = B;
    TopEnd = E;
    TopFA = &I->second;
    if (TopEnd >= EndOffs)
      return;
    unsigned diff = EndOffs.getOffset() - TopEnd.getOffset();
    TopEnd = EndOffs;
    TopFA->RemoveLen += diff;
    ++I;
  }

  while (I != FileEdits.end()) {
    FileEdit &FA = I->second;
    FileOffset B = I->first;
    FileOffset E = B.getWithOffset(FA.RemoveLen);

    if (B >= TopEnd)
      break;

    if (E <= TopEnd) {
      FileEdits.erase(I++);
      continue;
    }

    if (B < TopEnd) {
      unsigned diff = E.getOffset() - TopEnd.getOffset();
      TopEnd = E;
      TopFA->RemoveLen += diff;
      FileEdits.erase(I);
    }

    break;
  }
}

bool EditedSource::commit(const Commit &commit) {
  if (!commit.isCommitable())
    return false;

  for (edit::Commit::edit_iterator
         I = commit.edit_begin(), E = commit.edit_end(); I != E; ++I) {
    const edit::Commit::Edit &edit = *I;
    switch (edit.Kind) {
    case edit::Commit::Act_Insert:
      commitInsert(edit.OrigLoc, edit.Offset, edit.Text, edit.BeforePrev);
      break;
    case edit::Commit::Act_InsertFromRange:
      commitInsertFromRange(edit.OrigLoc, edit.Offset,
                            edit.InsertFromRangeOffs, edit.Length,
                            edit.BeforePrev);
      break;
    case edit::Commit::Act_Remove:
      commitRemove(edit.OrigLoc, edit.Offset, edit.Length);
      break;
    }
  }

  return true;
}

static void applyRewrite(EditsReceiver &receiver,
                         StringRef text, FileOffset offs, unsigned len,
                         const SourceManager &SM) {
  assert(!offs.getFID().isInvalid());
  SourceLocation Loc = SM.getLocForStartOfFile(offs.getFID());
  Loc = Loc.getLocWithOffset(offs.getOffset());
  assert(Loc.isFileID());
  CharSourceRange range = CharSourceRange::getCharRange(Loc,
                                                     Loc.getLocWithOffset(len));

  if (text.empty()) {
    assert(len);
    receiver.remove(range);
    return;
  }

  if (len)
    receiver.replace(range, text);
  else
    receiver.insert(Loc, text);
}

void EditedSource::applyRewrites(EditsReceiver &receiver) {
  llvm::SmallString<128> StrVec;
  FileOffset CurOffs, CurEnd;
  unsigned CurLen;

  if (FileEdits.empty())
    return;

  FileEditsTy::iterator I = FileEdits.begin();
  CurOffs = I->first;
  StrVec = I->second.Text;
  CurLen = I->second.RemoveLen;
  CurEnd = CurOffs.getWithOffset(CurLen);
  ++I;

  for (FileEditsTy::iterator E = FileEdits.end(); I != E; ++I) {
    FileOffset offs = I->first;
    FileEdit act = I->second;
    assert(offs >= CurEnd);

    if (offs == CurEnd) {
      StrVec += act.Text;
      CurLen += act.RemoveLen;
      CurEnd.getWithOffset(act.RemoveLen);
      continue;
    }

    applyRewrite(receiver, StrVec.str(), CurOffs, CurLen, SourceMgr);
    CurOffs = offs;
    StrVec = act.Text;
    CurLen = act.RemoveLen;
    CurEnd = CurOffs.getWithOffset(CurLen);
  }

  applyRewrite(receiver, StrVec.str(), CurOffs, CurLen, SourceMgr);
}

void EditedSource::clearRewrites() {
  FileEdits.clear();
  StrAlloc.Reset();
}

StringRef EditedSource::getSourceText(FileOffset BeginOffs, FileOffset EndOffs,
                                      bool &Invalid) {
  assert(BeginOffs.getFID() == EndOffs.getFID());
  assert(BeginOffs <= EndOffs);
  SourceLocation BLoc = SourceMgr.getLocForStartOfFile(BeginOffs.getFID());
  BLoc = BLoc.getLocWithOffset(BeginOffs.getOffset());
  assert(BLoc.isFileID());
  SourceLocation
    ELoc = BLoc.getLocWithOffset(EndOffs.getOffset() - BeginOffs.getOffset());
  return Lexer::getSourceText(CharSourceRange::getCharRange(BLoc, ELoc),
                              SourceMgr, LangOpts, &Invalid);
}

EditedSource::FileEditsTy::iterator
EditedSource::getActionForOffset(FileOffset Offs) {
  FileEditsTy::iterator I = FileEdits.upper_bound(Offs);
  if (I == FileEdits.begin())
    return FileEdits.end();
  --I;
  FileEdit &FA = I->second;
  FileOffset B = I->first;
  FileOffset E = B.getWithOffset(FA.RemoveLen);
  if (Offs >= B && Offs < E)
    return I;

  return FileEdits.end();
}
