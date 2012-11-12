//===--- RawCommentList.cpp - Processing raw comments -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RawCommentList.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/CommentBriefParser.h"
#include "clang/AST/CommentSema.h"
#include "clang/AST/CommentParser.h"
#include "clang/AST/CommentCommandTraits.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;

namespace {
/// Get comment kind and bool describing if it is a trailing comment.
std::pair<RawComment::CommentKind, bool> getCommentKind(StringRef Comment) {
  if (Comment.size() < 3 || Comment[0] != '/')
    return std::make_pair(RawComment::RCK_Invalid, false);

  RawComment::CommentKind K;
  if (Comment[1] == '/') {
    if (Comment.size() < 3)
      return std::make_pair(RawComment::RCK_OrdinaryBCPL, false);

    if (Comment[2] == '/')
      K = RawComment::RCK_BCPLSlash;
    else if (Comment[2] == '!')
      K = RawComment::RCK_BCPLExcl;
    else
      return std::make_pair(RawComment::RCK_OrdinaryBCPL, false);
  } else {
    assert(Comment.size() >= 4);

    // Comment lexer does not understand escapes in comment markers, so pretend
    // that this is not a comment.
    if (Comment[1] != '*' ||
        Comment[Comment.size() - 2] != '*' ||
        Comment[Comment.size() - 1] != '/')
      return std::make_pair(RawComment::RCK_Invalid, false);

    if (Comment[2] == '*')
      K = RawComment::RCK_JavaDoc;
    else if (Comment[2] == '!')
      K = RawComment::RCK_Qt;
    else
      return std::make_pair(RawComment::RCK_OrdinaryC, false);
  }
  const bool TrailingComment = (Comment.size() > 3) && (Comment[3] == '<');
  return std::make_pair(K, TrailingComment);
}

bool mergedCommentIsTrailingComment(StringRef Comment) {
  return (Comment.size() > 3) && (Comment[3] == '<');
}
} // unnamed namespace

RawComment::RawComment(const SourceManager &SourceMgr, SourceRange SR,
                       bool Merged) :
    Range(SR), RawTextValid(false), BriefTextValid(false),
    IsAttached(false), IsAlmostTrailingComment(false),
    BeginLineValid(false), EndLineValid(false) {
  // Extract raw comment text, if possible.
  if (SR.getBegin() == SR.getEnd() || getRawText(SourceMgr).empty()) {
    Kind = RCK_Invalid;
    return;
  }

  if (!Merged) {
    // Guess comment kind.
    std::pair<CommentKind, bool> K = getCommentKind(RawText);
    Kind = K.first;
    IsTrailingComment = K.second;

    IsAlmostTrailingComment = RawText.startswith("//<") ||
                                 RawText.startswith("/*<");
  } else {
    Kind = RCK_Merged;
    IsTrailingComment = mergedCommentIsTrailingComment(RawText);
  }
}

unsigned RawComment::getBeginLine(const SourceManager &SM) const {
  if (BeginLineValid)
    return BeginLine;

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getBegin());
  BeginLine = SM.getLineNumber(LocInfo.first, LocInfo.second);
  BeginLineValid = true;
  return BeginLine;
}

unsigned RawComment::getEndLine(const SourceManager &SM) const {
  if (EndLineValid)
    return EndLine;

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getEnd());
  EndLine = SM.getLineNumber(LocInfo.first, LocInfo.second);
  EndLineValid = true;
  return EndLine;
}

StringRef RawComment::getRawTextSlow(const SourceManager &SourceMgr) const {
  FileID BeginFileID;
  FileID EndFileID;
  unsigned BeginOffset;
  unsigned EndOffset;

  llvm::tie(BeginFileID, BeginOffset) =
      SourceMgr.getDecomposedLoc(Range.getBegin());
  llvm::tie(EndFileID, EndOffset) =
      SourceMgr.getDecomposedLoc(Range.getEnd());

  const unsigned Length = EndOffset - BeginOffset;
  if (Length < 2)
    return StringRef();

  // The comment can't begin in one file and end in another.
  assert(BeginFileID == EndFileID);

  bool Invalid = false;
  const char *BufferStart = SourceMgr.getBufferData(BeginFileID,
                                                    &Invalid).data();
  if (Invalid)
    return StringRef();

  return StringRef(BufferStart + BeginOffset, Length);
}

const char *RawComment::extractBriefText(const ASTContext &Context) const {
  // Make sure that RawText is valid.
  getRawText(Context.getSourceManager());

  // Since we will be copying the resulting text, all allocations made during
  // parsing are garbage after resulting string is formed.  Thus we can use
  // a separate allocator for all temporary stuff.
  llvm::BumpPtrAllocator Allocator;

  comments::Lexer L(Allocator, Context.getCommentCommandTraits(),
                    Range.getBegin(),
                    RawText.begin(), RawText.end());
  comments::BriefParser P(L, Context.getCommentCommandTraits());

  const std::string Result = P.Parse();
  const unsigned BriefTextLength = Result.size();
  char *BriefTextPtr = new (Context) char[BriefTextLength + 1];
  memcpy(BriefTextPtr, Result.c_str(), BriefTextLength + 1);
  BriefText = BriefTextPtr;
  BriefTextValid = true;

  return BriefTextPtr;
}

comments::FullComment *RawComment::parse(const ASTContext &Context,
                                         const Preprocessor *PP,
                                         const Decl *D) const {
  // Make sure that RawText is valid.
  getRawText(Context.getSourceManager());

  comments::Lexer L(Context.getAllocator(), Context.getCommentCommandTraits(),
                    getSourceRange().getBegin(),
                    RawText.begin(), RawText.end());
  comments::Sema S(Context.getAllocator(), Context.getSourceManager(),
                   Context.getDiagnostics(),
                   Context.getCommentCommandTraits(),
                   PP);
  S.setDecl(D);
  comments::Parser P(L, S, Context.getAllocator(), Context.getSourceManager(),
                     Context.getDiagnostics(),
                     Context.getCommentCommandTraits());

  return P.parseFullComment();
}

namespace {
bool containsOnlyWhitespace(StringRef Str) {
  return Str.find_first_not_of(" \t\f\v\r\n") == StringRef::npos;
}

bool onlyWhitespaceBetween(SourceManager &SM,
                           SourceLocation Loc1, SourceLocation Loc2) {
  std::pair<FileID, unsigned> Loc1Info = SM.getDecomposedLoc(Loc1);
  std::pair<FileID, unsigned> Loc2Info = SM.getDecomposedLoc(Loc2);

  // Question does not make sense if locations are in different files.
  if (Loc1Info.first != Loc2Info.first)
    return false;

  bool Invalid = false;
  const char *Buffer = SM.getBufferData(Loc1Info.first, &Invalid).data();
  if (Invalid)
    return false;

  StringRef Text(Buffer + Loc1Info.second, Loc2Info.second - Loc1Info.second);
  return containsOnlyWhitespace(Text);
}
} // unnamed namespace

void RawCommentList::addComment(const RawComment &RC,
                                llvm::BumpPtrAllocator &Allocator) {
  if (RC.isInvalid())
    return;

  // Check if the comments are not in source order.
  while (!Comments.empty() &&
         !SourceMgr.isBeforeInTranslationUnit(
              Comments.back()->getSourceRange().getBegin(),
              RC.getSourceRange().getBegin())) {
    // If they are, just pop a few last comments that don't fit.
    // This happens if an \#include directive contains comments.
    Comments.pop_back();
  }

  if (OnlyWhitespaceSeen) {
    if (!onlyWhitespaceBetween(SourceMgr,
                               PrevCommentEndLoc,
                               RC.getSourceRange().getBegin()))
      OnlyWhitespaceSeen = false;
  }

  PrevCommentEndLoc = RC.getSourceRange().getEnd();

  // Ordinary comments are not interesting for us.
  if (RC.isOrdinary())
    return;

  // If this is the first Doxygen comment, save it (because there isn't
  // anything to merge it with).
  if (Comments.empty()) {
    Comments.push_back(new (Allocator) RawComment(RC));
    OnlyWhitespaceSeen = true;
    return;
  }

  const RawComment &C1 = *Comments.back();
  const RawComment &C2 = RC;

  // Merge comments only if there is only whitespace between them.
  // Can't merge trailing and non-trailing comments.
  // Merge comments if they are on same or consecutive lines.
  bool Merged = false;
  if (OnlyWhitespaceSeen &&
      (C1.isTrailingComment() == C2.isTrailingComment())) {
    unsigned C1EndLine = C1.getEndLine(SourceMgr);
    unsigned C2BeginLine = C2.getBeginLine(SourceMgr);
    if (C1EndLine + 1 == C2BeginLine || C1EndLine == C2BeginLine) {
      SourceRange MergedRange(C1.getSourceRange().getBegin(),
                              C2.getSourceRange().getEnd());
      *Comments.back() = RawComment(SourceMgr, MergedRange, true);
      Merged = true;
    }
  }
  if (!Merged)
    Comments.push_back(new (Allocator) RawComment(RC));

  OnlyWhitespaceSeen = true;
}

