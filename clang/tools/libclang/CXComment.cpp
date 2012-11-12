//===- CXComment.cpp - libclang APIs for manipulating CXComments ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all libclang APIs related to walking comment AST.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "CXString.h"
#include "CXComment.h"
#include "CXCursor.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/Decl.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>

using namespace clang;
using namespace clang::cxstring;
using namespace clang::comments;
using namespace clang::cxcomment;

extern "C" {

enum CXCommentKind clang_Comment_getKind(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return CXComment_Null;

  switch (C->getCommentKind()) {
  case Comment::NoCommentKind:
    return CXComment_Null;

  case Comment::TextCommentKind:
    return CXComment_Text;

  case Comment::InlineCommandCommentKind:
    return CXComment_InlineCommand;

  case Comment::HTMLStartTagCommentKind:
    return CXComment_HTMLStartTag;

  case Comment::HTMLEndTagCommentKind:
    return CXComment_HTMLEndTag;

  case Comment::ParagraphCommentKind:
    return CXComment_Paragraph;

  case Comment::BlockCommandCommentKind:
    return CXComment_BlockCommand;

  case Comment::ParamCommandCommentKind:
    return CXComment_ParamCommand;

  case Comment::TParamCommandCommentKind:
    return CXComment_TParamCommand;

  case Comment::VerbatimBlockCommentKind:
    return CXComment_VerbatimBlockCommand;

  case Comment::VerbatimBlockLineCommentKind:
    return CXComment_VerbatimBlockLine;

  case Comment::VerbatimLineCommentKind:
    return CXComment_VerbatimLine;

  case Comment::FullCommentKind:
    return CXComment_FullComment;
  }
  llvm_unreachable("unknown CommentKind");
}

unsigned clang_Comment_getNumChildren(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return 0;

  return C->child_count();
}

CXComment clang_Comment_getChild(CXComment CXC, unsigned ChildIdx) {
  const Comment *C = getASTNode(CXC);
  if (!C || ChildIdx >= C->child_count())
    return createCXComment(NULL, NULL);

  return createCXComment(*(C->child_begin() + ChildIdx), CXC.TranslationUnit);
}

unsigned clang_Comment_isWhitespace(CXComment CXC) {
  const Comment *C = getASTNode(CXC);
  if (!C)
    return false;

  if (const TextComment *TC = dyn_cast<TextComment>(C))
    return TC->isWhitespace();

  if (const ParagraphComment *PC = dyn_cast<ParagraphComment>(C))
    return PC->isWhitespace();

  return false;
}

unsigned clang_InlineContentComment_hasTrailingNewline(CXComment CXC) {
  const InlineContentComment *ICC = getASTNodeAs<InlineContentComment>(CXC);
  if (!ICC)
    return false;

  return ICC->hasTrailingNewline();
}

CXString clang_TextComment_getText(CXComment CXC) {
  const TextComment *TC = getASTNodeAs<TextComment>(CXC);
  if (!TC)
    return createCXString((const char *) 0);

  return createCXString(TC->getText(), /*DupString=*/ false);
}

CXString clang_InlineCommandComment_getCommandName(CXComment CXC) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC)
    return createCXString((const char *) 0);

  const CommandTraits &Traits = getCommandTraits(CXC);
  return createCXString(ICC->getCommandName(Traits), /*DupString=*/ false);
}

enum CXCommentInlineCommandRenderKind
clang_InlineCommandComment_getRenderKind(CXComment CXC) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC)
    return CXCommentInlineCommandRenderKind_Normal;

  switch (ICC->getRenderKind()) {
  case InlineCommandComment::RenderNormal:
    return CXCommentInlineCommandRenderKind_Normal;

  case InlineCommandComment::RenderBold:
    return CXCommentInlineCommandRenderKind_Bold;

  case InlineCommandComment::RenderMonospaced:
    return CXCommentInlineCommandRenderKind_Monospaced;

  case InlineCommandComment::RenderEmphasized:
    return CXCommentInlineCommandRenderKind_Emphasized;
  }
  llvm_unreachable("unknown InlineCommandComment::RenderKind");
}

unsigned clang_InlineCommandComment_getNumArgs(CXComment CXC) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC)
    return 0;

  return ICC->getNumArgs();
}

CXString clang_InlineCommandComment_getArgText(CXComment CXC,
                                               unsigned ArgIdx) {
  const InlineCommandComment *ICC = getASTNodeAs<InlineCommandComment>(CXC);
  if (!ICC || ArgIdx >= ICC->getNumArgs())
    return createCXString((const char *) 0);

  return createCXString(ICC->getArgText(ArgIdx), /*DupString=*/ false);
}

CXString clang_HTMLTagComment_getTagName(CXComment CXC) {
  const HTMLTagComment *HTC = getASTNodeAs<HTMLTagComment>(CXC);
  if (!HTC)
    return createCXString((const char *) 0);

  return createCXString(HTC->getTagName(), /*DupString=*/ false);
}

unsigned clang_HTMLStartTagComment_isSelfClosing(CXComment CXC) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST)
    return false;

  return HST->isSelfClosing();
}

unsigned clang_HTMLStartTag_getNumAttrs(CXComment CXC) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST)
    return 0;

  return HST->getNumAttrs();
}

CXString clang_HTMLStartTag_getAttrName(CXComment CXC, unsigned AttrIdx) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST || AttrIdx >= HST->getNumAttrs())
    return createCXString((const char *) 0);

  return createCXString(HST->getAttr(AttrIdx).Name, /*DupString=*/ false);
}

CXString clang_HTMLStartTag_getAttrValue(CXComment CXC, unsigned AttrIdx) {
  const HTMLStartTagComment *HST = getASTNodeAs<HTMLStartTagComment>(CXC);
  if (!HST || AttrIdx >= HST->getNumAttrs())
    return createCXString((const char *) 0);

  return createCXString(HST->getAttr(AttrIdx).Value, /*DupString=*/ false);
}

CXString clang_BlockCommandComment_getCommandName(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return createCXString((const char *) 0);

  const CommandTraits &Traits = getCommandTraits(CXC);
  return createCXString(BCC->getCommandName(Traits), /*DupString=*/ false);
}

unsigned clang_BlockCommandComment_getNumArgs(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return 0;

  return BCC->getNumArgs();
}

CXString clang_BlockCommandComment_getArgText(CXComment CXC,
                                              unsigned ArgIdx) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC || ArgIdx >= BCC->getNumArgs())
    return createCXString((const char *) 0);

  return createCXString(BCC->getArgText(ArgIdx), /*DupString=*/ false);
}

CXComment clang_BlockCommandComment_getParagraph(CXComment CXC) {
  const BlockCommandComment *BCC = getASTNodeAs<BlockCommandComment>(CXC);
  if (!BCC)
    return createCXComment(NULL, NULL);

  return createCXComment(BCC->getParagraph(), CXC.TranslationUnit);
}

CXString clang_ParamCommandComment_getParamName(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC || !PCC->hasParamName())
    return createCXString((const char *) 0);

  return createCXString(PCC->getParamNameAsWritten(), /*DupString=*/ false);
}

unsigned clang_ParamCommandComment_isParamIndexValid(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return false;

  return PCC->isParamIndexValid();
}

unsigned clang_ParamCommandComment_getParamIndex(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC || !PCC->isParamIndexValid())
    return ParamCommandComment::InvalidParamIndex;

  return PCC->getParamIndex();
}

unsigned clang_ParamCommandComment_isDirectionExplicit(CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return false;

  return PCC->isDirectionExplicit();
}

enum CXCommentParamPassDirection clang_ParamCommandComment_getDirection(
                                                            CXComment CXC) {
  const ParamCommandComment *PCC = getASTNodeAs<ParamCommandComment>(CXC);
  if (!PCC)
    return CXCommentParamPassDirection_In;

  switch (PCC->getDirection()) {
  case ParamCommandComment::In:
    return CXCommentParamPassDirection_In;

  case ParamCommandComment::Out:
    return CXCommentParamPassDirection_Out;

  case ParamCommandComment::InOut:
    return CXCommentParamPassDirection_InOut;
  }
  llvm_unreachable("unknown ParamCommandComment::PassDirection");
}

CXString clang_TParamCommandComment_getParamName(CXComment CXC) {
  const TParamCommandComment *TPCC = getASTNodeAs<TParamCommandComment>(CXC);
  if (!TPCC || !TPCC->hasParamName())
    return createCXString((const char *) 0);

  return createCXString(TPCC->getParamNameAsWritten(), /*DupString=*/ false);
}

unsigned clang_TParamCommandComment_isParamPositionValid(CXComment CXC) {
  const TParamCommandComment *TPCC = getASTNodeAs<TParamCommandComment>(CXC);
  if (!TPCC)
    return false;

  return TPCC->isPositionValid();
}

unsigned clang_TParamCommandComment_getDepth(CXComment CXC) {
  const TParamCommandComment *TPCC = getASTNodeAs<TParamCommandComment>(CXC);
  if (!TPCC || !TPCC->isPositionValid())
    return 0;

  return TPCC->getDepth();
}

unsigned clang_TParamCommandComment_getIndex(CXComment CXC, unsigned Depth) {
  const TParamCommandComment *TPCC = getASTNodeAs<TParamCommandComment>(CXC);
  if (!TPCC || !TPCC->isPositionValid() || Depth >= TPCC->getDepth())
    return 0;

  return TPCC->getIndex(Depth);
}

CXString clang_VerbatimBlockLineComment_getText(CXComment CXC) {
  const VerbatimBlockLineComment *VBL =
      getASTNodeAs<VerbatimBlockLineComment>(CXC);
  if (!VBL)
    return createCXString((const char *) 0);

  return createCXString(VBL->getText(), /*DupString=*/ false);
}

CXString clang_VerbatimLineComment_getText(CXComment CXC) {
  const VerbatimLineComment *VLC = getASTNodeAs<VerbatimLineComment>(CXC);
  if (!VLC)
    return createCXString((const char *) 0);

  return createCXString(VLC->getText(), /*DupString=*/ false);
}

} // end extern "C"

//===----------------------------------------------------------------------===//
// Helpers for converting comment AST to HTML.
//===----------------------------------------------------------------------===//

namespace {

/// This comparison will sort parameters with valid index by index and
/// invalid (unresolved) parameters last.
class ParamCommandCommentCompareIndex {
public:
  bool operator()(const ParamCommandComment *LHS,
                  const ParamCommandComment *RHS) const {
    unsigned LHSIndex = UINT_MAX;
    unsigned RHSIndex = UINT_MAX;
    if (LHS->isParamIndexValid())
      LHSIndex = LHS->getParamIndex();
    if (RHS->isParamIndexValid())
      RHSIndex = RHS->getParamIndex();

    return LHSIndex < RHSIndex;
  }
};

/// This comparison will sort template parameters in the following order:
/// \li real template parameters (depth = 1) in index order;
/// \li all other names (depth > 1);
/// \li unresolved names.
class TParamCommandCommentComparePosition {
public:
  bool operator()(const TParamCommandComment *LHS,
                  const TParamCommandComment *RHS) const {
    // Sort unresolved names last.
    if (!LHS->isPositionValid())
      return false;
    if (!RHS->isPositionValid())
      return true;

    if (LHS->getDepth() > 1)
      return false;
    if (RHS->getDepth() > 1)
      return true;

    // Sort template parameters in index order.
    if (LHS->getDepth() == 1 && RHS->getDepth() == 1)
      return LHS->getIndex(0) < RHS->getIndex(0);

    // Leave all other names in source order.
    return true;
  }
};

/// Separate parts of a FullComment.
struct FullCommentParts {
  /// Take a full comment apart and initialize members accordingly.
  FullCommentParts(const FullComment *C,
                   const CommandTraits &Traits);

  const BlockContentComment *Brief;
  const ParagraphComment *FirstParagraph;
  const BlockCommandComment *Returns;
  SmallVector<const ParamCommandComment *, 8> Params;
  SmallVector<const TParamCommandComment *, 4> TParams;
  SmallVector<const BlockContentComment *, 8> MiscBlocks;
};

FullCommentParts::FullCommentParts(const FullComment *C,
                                   const CommandTraits &Traits) :
    Brief(NULL), FirstParagraph(NULL), Returns(NULL) {
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    const Comment *Child = *I;
    if (!Child)
      continue;
    switch (Child->getCommentKind()) {
    case Comment::NoCommentKind:
      continue;

    case Comment::ParagraphCommentKind: {
      const ParagraphComment *PC = cast<ParagraphComment>(Child);
      if (PC->isWhitespace())
        break;
      if (!FirstParagraph)
        FirstParagraph = PC;

      MiscBlocks.push_back(PC);
      break;
    }

    case Comment::BlockCommandCommentKind: {
      const BlockCommandComment *BCC = cast<BlockCommandComment>(Child);
      const CommandInfo *Info = Traits.getCommandInfo(BCC->getCommandID());
      if (!Brief && Info->IsBriefCommand) {
        Brief = BCC;
        break;
      }
      if (!Returns && Info->IsReturnsCommand) {
        Returns = BCC;
        break;
      }
      MiscBlocks.push_back(BCC);
      break;
    }

    case Comment::ParamCommandCommentKind: {
      const ParamCommandComment *PCC = cast<ParamCommandComment>(Child);
      if (!PCC->hasParamName())
        break;

      if (!PCC->isDirectionExplicit() && !PCC->hasNonWhitespaceParagraph())
        break;

      Params.push_back(PCC);
      break;
    }

    case Comment::TParamCommandCommentKind: {
      const TParamCommandComment *TPCC = cast<TParamCommandComment>(Child);
      if (!TPCC->hasParamName())
        break;

      if (!TPCC->hasNonWhitespaceParagraph())
        break;

      TParams.push_back(TPCC);
      break;
    }

    case Comment::VerbatimBlockCommentKind:
      MiscBlocks.push_back(cast<BlockCommandComment>(Child));
      break;

    case Comment::VerbatimLineCommentKind: {
      const VerbatimLineComment *VLC = cast<VerbatimLineComment>(Child);
      const CommandInfo *Info = Traits.getCommandInfo(VLC->getCommandID());
      if (!Info->IsDeclarationCommand)
        MiscBlocks.push_back(VLC);
      break;
    }

    case Comment::TextCommentKind:
    case Comment::InlineCommandCommentKind:
    case Comment::HTMLStartTagCommentKind:
    case Comment::HTMLEndTagCommentKind:
    case Comment::VerbatimBlockLineCommentKind:
    case Comment::FullCommentKind:
      llvm_unreachable("AST node of this kind can't be a child of "
                       "a FullComment");
    }
  }

  // Sort params in order they are declared in the function prototype.
  // Unresolved parameters are put at the end of the list in the same order
  // they were seen in the comment.
  std::stable_sort(Params.begin(), Params.end(),
                   ParamCommandCommentCompareIndex());

  std::stable_sort(TParams.begin(), TParams.end(),
                   TParamCommandCommentComparePosition());
}

void PrintHTMLStartTagComment(const HTMLStartTagComment *C,
                              llvm::raw_svector_ostream &Result) {
  Result << "<" << C->getTagName();

  if (C->getNumAttrs() != 0) {
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; i++) {
      Result << " ";
      const HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      Result << Attr.Name;
      if (!Attr.Value.empty())
        Result << "=\"" << Attr.Value << "\"";
    }
  }

  if (!C->isSelfClosing())
    Result << ">";
  else
    Result << "/>";
}

class CommentASTToHTMLConverter :
    public ConstCommentVisitor<CommentASTToHTMLConverter> {
public:
  /// \param Str accumulator for HTML.
  CommentASTToHTMLConverter(const FullComment *FC,
                            SmallVectorImpl<char> &Str,
                            const CommandTraits &Traits) :
      FC(FC), Result(Str), Traits(Traits)
  { }

  // Inline content.
  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLStartTagComment(const HTMLStartTagComment *C);
  void visitHTMLEndTagComment(const HTMLEndTagComment *C);

  // Block content.
  void visitParagraphComment(const ParagraphComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitTParamCommandComment(const TParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

  void visitFullComment(const FullComment *C);

  // Helpers.

  /// Convert a paragraph that is not a block by itself (an argument to some
  /// command).
  void visitNonStandaloneParagraphComment(const ParagraphComment *C);

  void appendToResultWithHTMLEscaping(StringRef S);

private:
  const FullComment *FC;
  /// Output stream for HTML.
  llvm::raw_svector_ostream Result;

  const CommandTraits &Traits;
};
} // end unnamed namespace

void CommentASTToHTMLConverter::visitTextComment(const TextComment *C) {
  appendToResultWithHTMLEscaping(C->getText());
}

void CommentASTToHTMLConverter::visitInlineCommandComment(
                                  const InlineCommandComment *C) {
  // Nothing to render if no arguments supplied.
  if (C->getNumArgs() == 0)
    return;

  // Nothing to render if argument is empty.
  StringRef Arg0 = C->getArgText(0);
  if (Arg0.empty())
    return;

  switch (C->getRenderKind()) {
  case InlineCommandComment::RenderNormal:
    for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i) {
      appendToResultWithHTMLEscaping(C->getArgText(i));
      Result << " ";
    }
    return;

  case InlineCommandComment::RenderBold:
    assert(C->getNumArgs() == 1);
    Result << "<b>";
    appendToResultWithHTMLEscaping(Arg0);
    Result << "</b>";
    return;
  case InlineCommandComment::RenderMonospaced:
    assert(C->getNumArgs() == 1);
    Result << "<tt>";
    appendToResultWithHTMLEscaping(Arg0);
    Result<< "</tt>";
    return;
  case InlineCommandComment::RenderEmphasized:
    assert(C->getNumArgs() == 1);
    Result << "<em>";
    appendToResultWithHTMLEscaping(Arg0);
    Result << "</em>";
    return;
  }
}

void CommentASTToHTMLConverter::visitHTMLStartTagComment(
                                  const HTMLStartTagComment *C) {
  PrintHTMLStartTagComment(C, Result);
}

void CommentASTToHTMLConverter::visitHTMLEndTagComment(
                                  const HTMLEndTagComment *C) {
  Result << "</" << C->getTagName() << ">";
}

void CommentASTToHTMLConverter::visitParagraphComment(
                                  const ParagraphComment *C) {
  if (C->isWhitespace())
    return;

  Result << "<p>";
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    visit(*I);
  }
  Result << "</p>";
}

void CommentASTToHTMLConverter::visitBlockCommandComment(
                                  const BlockCommandComment *C) {
  const CommandInfo *Info = Traits.getCommandInfo(C->getCommandID());
  if (Info->IsBriefCommand) {
    Result << "<p class=\"para-brief\">";
    visitNonStandaloneParagraphComment(C->getParagraph());
    Result << "</p>";
    return;
  }
  if (Info->IsReturnsCommand) {
    Result << "<p class=\"para-returns\">"
              "<span class=\"word-returns\">Returns</span> ";
    visitNonStandaloneParagraphComment(C->getParagraph());
    Result << "</p>";
    return;
  }
  // We don't know anything about this command.  Just render the paragraph.
  visit(C->getParagraph());
}

void CommentASTToHTMLConverter::visitParamCommandComment(
                                  const ParamCommandComment *C) {
  if (C->isParamIndexValid()) {
    Result << "<dt class=\"param-name-index-"
           << C->getParamIndex()
           << "\">";
    appendToResultWithHTMLEscaping(C->getParamName(FC));
  } else {
    Result << "<dt class=\"param-name-index-invalid\">";
    appendToResultWithHTMLEscaping(C->getParamNameAsWritten());
  }
  Result << "</dt>";

  if (C->isParamIndexValid()) {
    Result << "<dd class=\"param-descr-index-"
           << C->getParamIndex()
           << "\">";
  } else
    Result << "<dd class=\"param-descr-index-invalid\">";

  visitNonStandaloneParagraphComment(C->getParagraph());
  Result << "</dd>";
}

void CommentASTToHTMLConverter::visitTParamCommandComment(
                                  const TParamCommandComment *C) {
  if (C->isPositionValid()) {
    if (C->getDepth() == 1)
      Result << "<dt class=\"tparam-name-index-"
             << C->getIndex(0)
             << "\">";
    else
      Result << "<dt class=\"tparam-name-index-other\">";
    appendToResultWithHTMLEscaping(C->getParamName(FC));
  } else {
    Result << "<dt class=\"tparam-name-index-invalid\">";
    appendToResultWithHTMLEscaping(C->getParamNameAsWritten());
  }
  
  Result << "</dt>";

  if (C->isPositionValid()) {
    if (C->getDepth() == 1)
      Result << "<dd class=\"tparam-descr-index-"
             << C->getIndex(0)
             << "\">";
    else
      Result << "<dd class=\"tparam-descr-index-other\">";
  } else
    Result << "<dd class=\"tparam-descr-index-invalid\">";

  visitNonStandaloneParagraphComment(C->getParagraph());
  Result << "</dd>";
}

void CommentASTToHTMLConverter::visitVerbatimBlockComment(
                                  const VerbatimBlockComment *C) {
  unsigned NumLines = C->getNumLines();
  if (NumLines == 0)
    return;

  Result << "<pre>";
  for (unsigned i = 0; i != NumLines; ++i) {
    appendToResultWithHTMLEscaping(C->getText(i));
    if (i + 1 != NumLines)
      Result << '\n';
  }
  Result << "</pre>";
}

void CommentASTToHTMLConverter::visitVerbatimBlockLineComment(
                                  const VerbatimBlockLineComment *C) {
  llvm_unreachable("should not see this AST node");
}

void CommentASTToHTMLConverter::visitVerbatimLineComment(
                                  const VerbatimLineComment *C) {
  Result << "<pre>";
  appendToResultWithHTMLEscaping(C->getText());
  Result << "</pre>";
}

void CommentASTToHTMLConverter::visitFullComment(const FullComment *C) {
  FullCommentParts Parts(C, Traits);

  bool FirstParagraphIsBrief = false;
  if (Parts.Brief)
    visit(Parts.Brief);
  else if (Parts.FirstParagraph) {
    Result << "<p class=\"para-brief\">";
    visitNonStandaloneParagraphComment(Parts.FirstParagraph);
    Result << "</p>";
    FirstParagraphIsBrief = true;
  }

  for (unsigned i = 0, e = Parts.MiscBlocks.size(); i != e; ++i) {
    const Comment *C = Parts.MiscBlocks[i];
    if (FirstParagraphIsBrief && C == Parts.FirstParagraph)
      continue;
    visit(C);
  }

  if (Parts.TParams.size() != 0) {
    Result << "<dl>";
    for (unsigned i = 0, e = Parts.TParams.size(); i != e; ++i)
      visit(Parts.TParams[i]);
    Result << "</dl>";
  }

  if (Parts.Params.size() != 0) {
    Result << "<dl>";
    for (unsigned i = 0, e = Parts.Params.size(); i != e; ++i)
      visit(Parts.Params[i]);
    Result << "</dl>";
  }

  if (Parts.Returns)
    visit(Parts.Returns);

  Result.flush();
}

void CommentASTToHTMLConverter::visitNonStandaloneParagraphComment(
                                  const ParagraphComment *C) {
  if (!C)
    return;

  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    visit(*I);
  }
}

void CommentASTToHTMLConverter::appendToResultWithHTMLEscaping(StringRef S) {
  for (StringRef::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    const char C = *I;
    switch (C) {
      case '&':
        Result << "&amp;";
        break;
      case '<':
        Result << "&lt;";
        break;
      case '>':
        Result << "&gt;";
        break;
      case '"':
        Result << "&quot;";
        break;
      case '\'':
        Result << "&#39;";
        break;
      case '/':
        Result << "&#47;";
        break;
      default:
        Result << C;
        break;
    }
  }
}

extern "C" {

CXString clang_HTMLTagComment_getAsString(CXComment CXC) {
  const HTMLTagComment *HTC = getASTNodeAs<HTMLTagComment>(CXC);
  if (!HTC)
    return createCXString((const char *) 0);

  SmallString<128> HTML;
  CommentASTToHTMLConverter Converter(0, HTML, getCommandTraits(CXC));
  Converter.visit(HTC);
  return createCXString(HTML.str(), /* DupString = */ true);
}

CXString clang_FullComment_getAsHTML(CXComment CXC) {
  const FullComment *FC = getASTNodeAs<FullComment>(CXC);
  if (!FC)
    return createCXString((const char *) 0);

  SmallString<1024> HTML;
  CommentASTToHTMLConverter Converter(FC, HTML, getCommandTraits(CXC));
  Converter.visit(FC);
  return createCXString(HTML.str(), /* DupString = */ true);
}

} // end extern "C"

namespace {
class CommentASTToXMLConverter :
    public ConstCommentVisitor<CommentASTToXMLConverter> {
public:
  /// \param Str accumulator for XML.
  CommentASTToXMLConverter(const FullComment *FC,
                           SmallVectorImpl<char> &Str,
                           const CommandTraits &Traits,
                           const SourceManager &SM) :
      FC(FC), Result(Str), Traits(Traits), SM(SM) { }

  // Inline content.
  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLStartTagComment(const HTMLStartTagComment *C);
  void visitHTMLEndTagComment(const HTMLEndTagComment *C);

  // Block content.
  void visitParagraphComment(const ParagraphComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitTParamCommandComment(const TParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

  void visitFullComment(const FullComment *C);

  // Helpers.
  void appendToResultWithXMLEscaping(StringRef S);

private:
  const FullComment *FC;

  /// Output stream for XML.
  llvm::raw_svector_ostream Result;

  const CommandTraits &Traits;
  const SourceManager &SM;
};

void getSourceTextOfDeclaration(const DeclInfo *ThisDecl,
                                SmallVectorImpl<char> &Str) {
  ASTContext &Context = ThisDecl->CurrentDecl->getASTContext();
  const LangOptions &LangOpts = Context.getLangOpts();
  llvm::raw_svector_ostream OS(Str);
  PrintingPolicy PPolicy(LangOpts);
  PPolicy.SuppressAttributes = true;
  PPolicy.TerseOutput = true;
  ThisDecl->CurrentDecl->print(OS, PPolicy,
                               /*Indentation*/0, /*PrintInstantiation*/true);
}

} // end unnamed namespace

void CommentASTToXMLConverter::visitTextComment(const TextComment *C) {
  appendToResultWithXMLEscaping(C->getText());
}

void CommentASTToXMLConverter::visitInlineCommandComment(const InlineCommandComment *C) {
  // Nothing to render if no arguments supplied.
  if (C->getNumArgs() == 0)
    return;

  // Nothing to render if argument is empty.
  StringRef Arg0 = C->getArgText(0);
  if (Arg0.empty())
    return;

  switch (C->getRenderKind()) {
  case InlineCommandComment::RenderNormal:
    for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i) {
      appendToResultWithXMLEscaping(C->getArgText(i));
      Result << " ";
    }
    return;
  case InlineCommandComment::RenderBold:
    assert(C->getNumArgs() == 1);
    Result << "<bold>";
    appendToResultWithXMLEscaping(Arg0);
    Result << "</bold>";
    return;
  case InlineCommandComment::RenderMonospaced:
    assert(C->getNumArgs() == 1);
    Result << "<monospaced>";
    appendToResultWithXMLEscaping(Arg0);
    Result << "</monospaced>";
    return;
  case InlineCommandComment::RenderEmphasized:
    assert(C->getNumArgs() == 1);
    Result << "<emphasized>";
    appendToResultWithXMLEscaping(Arg0);
    Result << "</emphasized>";
    return;
  }
}

void CommentASTToXMLConverter::visitHTMLStartTagComment(const HTMLStartTagComment *C) {
  Result << "<rawHTML><![CDATA[";
  PrintHTMLStartTagComment(C, Result);
  Result << "]]></rawHTML>";
}

void CommentASTToXMLConverter::visitHTMLEndTagComment(const HTMLEndTagComment *C) {
  Result << "<rawHTML>&lt;/" << C->getTagName() << "&gt;</rawHTML>";
}

void CommentASTToXMLConverter::visitParagraphComment(const ParagraphComment *C) {
  if (C->isWhitespace())
    return;

  Result << "<Para>";
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    visit(*I);
  }
  Result << "</Para>";
}

void CommentASTToXMLConverter::visitBlockCommandComment(const BlockCommandComment *C) {
  visit(C->getParagraph());
}

void CommentASTToXMLConverter::visitParamCommandComment(const ParamCommandComment *C) {
  Result << "<Parameter><Name>";
  appendToResultWithXMLEscaping(C->isParamIndexValid() ? C->getParamName(FC)
                                                       : C->getParamNameAsWritten());
  Result << "</Name>";

  if (C->isParamIndexValid())
    Result << "<Index>" << C->getParamIndex() << "</Index>";

  Result << "<Direction isExplicit=\"" << C->isDirectionExplicit() << "\">";
  switch (C->getDirection()) {
  case ParamCommandComment::In:
    Result << "in";
    break;
  case ParamCommandComment::Out:
    Result << "out";
    break;
  case ParamCommandComment::InOut:
    Result << "in,out";
    break;
  }
  Result << "</Direction><Discussion>";
  visit(C->getParagraph());
  Result << "</Discussion></Parameter>";
}

void CommentASTToXMLConverter::visitTParamCommandComment(
                                  const TParamCommandComment *C) {
  Result << "<Parameter><Name>";
  appendToResultWithXMLEscaping(C->isPositionValid() ? C->getParamName(FC)
                                : C->getParamNameAsWritten());
  Result << "</Name>";

  if (C->isPositionValid() && C->getDepth() == 1) {
    Result << "<Index>" << C->getIndex(0) << "</Index>";
  }

  Result << "<Discussion>";
  visit(C->getParagraph());
  Result << "</Discussion></Parameter>";
}

void CommentASTToXMLConverter::visitVerbatimBlockComment(
                                  const VerbatimBlockComment *C) {
  unsigned NumLines = C->getNumLines();
  if (NumLines == 0)
    return;

  Result << llvm::StringSwitch<const char *>(C->getCommandName(Traits))
      .Case("code", "<Verbatim xml:space=\"preserve\" kind=\"code\">")
      .Default("<Verbatim xml:space=\"preserve\" kind=\"verbatim\">");
  for (unsigned i = 0; i != NumLines; ++i) {
    appendToResultWithXMLEscaping(C->getText(i));
    if (i + 1 != NumLines)
      Result << '\n';
  }
  Result << "</Verbatim>";
}

void CommentASTToXMLConverter::visitVerbatimBlockLineComment(
                                  const VerbatimBlockLineComment *C) {
  llvm_unreachable("should not see this AST node");
}

void CommentASTToXMLConverter::visitVerbatimLineComment(
                                  const VerbatimLineComment *C) {
  Result << "<Verbatim xml:space=\"preserve\" kind=\"verbatim\">";
  appendToResultWithXMLEscaping(C->getText());
  Result << "</Verbatim>";
}

void CommentASTToXMLConverter::visitFullComment(const FullComment *C) {
  FullCommentParts Parts(C, Traits);

  const DeclInfo *DI = C->getDeclInfo();
  StringRef RootEndTag;
  if (DI) {
    switch (DI->getKind()) {
    case DeclInfo::OtherKind:
      RootEndTag = "</Other>";
      Result << "<Other";
      break;
    case DeclInfo::FunctionKind:
      RootEndTag = "</Function>";
      Result << "<Function";
      switch (DI->TemplateKind) {
      case DeclInfo::NotTemplate:
        break;
      case DeclInfo::Template:
        Result << " templateKind=\"template\"";
        break;
      case DeclInfo::TemplateSpecialization:
        Result << " templateKind=\"specialization\"";
        break;
      case DeclInfo::TemplatePartialSpecialization:
        llvm_unreachable("partial specializations of functions "
                         "are not allowed in C++");
      }
      if (DI->IsInstanceMethod)
        Result << " isInstanceMethod=\"1\"";
      if (DI->IsClassMethod)
        Result << " isClassMethod=\"1\"";
      break;
    case DeclInfo::ClassKind:
      RootEndTag = "</Class>";
      Result << "<Class";
      switch (DI->TemplateKind) {
      case DeclInfo::NotTemplate:
        break;
      case DeclInfo::Template:
        Result << " templateKind=\"template\"";
        break;
      case DeclInfo::TemplateSpecialization:
        Result << " templateKind=\"specialization\"";
        break;
      case DeclInfo::TemplatePartialSpecialization:
        Result << " templateKind=\"partialSpecialization\"";
        break;
      }
      break;
    case DeclInfo::VariableKind:
      RootEndTag = "</Variable>";
      Result << "<Variable";
      break;
    case DeclInfo::NamespaceKind:
      RootEndTag = "</Namespace>";
      Result << "<Namespace";
      break;
    case DeclInfo::TypedefKind:
      RootEndTag = "</Typedef>";
      Result << "<Typedef";
      break;
    case DeclInfo::EnumKind:
      RootEndTag = "</Enum>";
      Result << "<Enum";
      break;
    }

    {
      // Print line and column number.
      SourceLocation Loc = DI->CurrentDecl->getLocation();
      std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
      FileID FID = LocInfo.first;
      unsigned FileOffset = LocInfo.second;

      if (!FID.isInvalid()) {
        if (const FileEntry *FE = SM.getFileEntryForID(FID)) {
          Result << " file=\"";
          appendToResultWithXMLEscaping(FE->getName());
          Result << "\"";
        }
        Result << " line=\"" << SM.getLineNumber(FID, FileOffset)
               << "\" column=\"" << SM.getColumnNumber(FID, FileOffset)
               << "\"";
      }
    }

    // Finish the root tag.
    Result << ">";

    bool FoundName = false;
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(DI->CommentDecl)) {
      if (DeclarationName DeclName = ND->getDeclName()) {
        Result << "<Name>";
        std::string Name = DeclName.getAsString();
        appendToResultWithXMLEscaping(Name);
        FoundName = true;
        Result << "</Name>";
      }
    }
    if (!FoundName)
      Result << "<Name>&lt;anonymous&gt;</Name>";

    {
      // Print USR.
      SmallString<128> USR;
      cxcursor::getDeclCursorUSR(DI->CommentDecl, USR);
      if (!USR.empty()) {
        Result << "<USR>";
        appendToResultWithXMLEscaping(USR);
        Result << "</USR>";
      }
    }
  } else {
    // No DeclInfo -- just emit some root tag and name tag.
    RootEndTag = "</Other>";
    Result << "<Other><Name>unknown</Name>";
  }

  {
    // Pretty-print the declaration.
    Result << "<Declaration>";
    SmallString<128> Declaration;
    getSourceTextOfDeclaration(DI, Declaration);
    appendToResultWithXMLEscaping(Declaration);
    Result << "</Declaration>";
  }

  bool FirstParagraphIsBrief = false;
  if (Parts.Brief) {
    Result << "<Abstract>";
    visit(Parts.Brief);
    Result << "</Abstract>";
  } else if (Parts.FirstParagraph) {
    Result << "<Abstract>";
    visit(Parts.FirstParagraph);
    Result << "</Abstract>";
    FirstParagraphIsBrief = true;
  }

  if (Parts.TParams.size() != 0) {
    Result << "<TemplateParameters>";
    for (unsigned i = 0, e = Parts.TParams.size(); i != e; ++i)
      visit(Parts.TParams[i]);
    Result << "</TemplateParameters>";
  }

  if (Parts.Params.size() != 0) {
    Result << "<Parameters>";
    for (unsigned i = 0, e = Parts.Params.size(); i != e; ++i)
      visit(Parts.Params[i]);
    Result << "</Parameters>";
  }

  if (Parts.Returns) {
    Result << "<ResultDiscussion>";
    visit(Parts.Returns);
    Result << "</ResultDiscussion>";
  }
  
  if (DI->CommentDecl->hasAttrs()) {
    const AttrVec &Attrs = DI->CommentDecl->getAttrs();
    for (unsigned i = 0, e = Attrs.size(); i != e; i++) {
      const AvailabilityAttr *AA = dyn_cast<AvailabilityAttr>(Attrs[i]);
      if (!AA) {
        if (const DeprecatedAttr *DA = dyn_cast<DeprecatedAttr>(Attrs[i])) {
          if (DA->getMessage().empty())
            Result << "<Deprecated/>";
          else {
            Result << "<Deprecated>";
            appendToResultWithXMLEscaping(DA->getMessage());
            Result << "</Deprecated>";
          }
        }
        else if (const UnavailableAttr *UA = dyn_cast<UnavailableAttr>(Attrs[i])) {
          if (UA->getMessage().empty())
            Result << "<Unavailable/>";
          else {
            Result << "<Unavailable>";
            appendToResultWithXMLEscaping(UA->getMessage());
            Result << "</Unavailable>";
          }
        }
        continue;
      }

      // 'availability' attribute.
      Result << "<Availability";
      StringRef Distribution;
      if (AA->getPlatform()) {
        Distribution = AvailabilityAttr::getPrettyPlatformName(
                                        AA->getPlatform()->getName());
        if (Distribution.empty())
          Distribution = AA->getPlatform()->getName();
      }
      Result << " distribution=\"" << Distribution << "\">";
      VersionTuple IntroducedInVersion = AA->getIntroduced();
      if (!IntroducedInVersion.empty()) {
        Result << "<IntroducedInVersion>"
               << IntroducedInVersion.getAsString()
               << "</IntroducedInVersion>";
      }
      VersionTuple DeprecatedInVersion = AA->getDeprecated();
      if (!DeprecatedInVersion.empty()) {
        Result << "<DeprecatedInVersion>"
               << DeprecatedInVersion.getAsString()
               << "</DeprecatedInVersion>";
      }
      VersionTuple RemovedAfterVersion = AA->getObsoleted();
      if (!RemovedAfterVersion.empty()) {
        Result << "<RemovedAfterVersion>"
               << RemovedAfterVersion.getAsString()
               << "</RemovedAfterVersion>";
      }
      StringRef DeprecationSummary = AA->getMessage();
      if (!DeprecationSummary.empty()) {
        Result << "<DeprecationSummary>";
        appendToResultWithXMLEscaping(DeprecationSummary);
        Result << "</DeprecationSummary>";
      }
      if (AA->getUnavailable())
        Result << "<Unavailable/>";
      Result << "</Availability>";
    }
  }

  {
    bool StartTagEmitted = false;
    for (unsigned i = 0, e = Parts.MiscBlocks.size(); i != e; ++i) {
      const Comment *C = Parts.MiscBlocks[i];
      if (FirstParagraphIsBrief && C == Parts.FirstParagraph)
        continue;
      if (!StartTagEmitted) {
        Result << "<Discussion>";
        StartTagEmitted = true;
      }
      visit(C);
    }
    if (StartTagEmitted)
      Result << "</Discussion>";
  }

  Result << RootEndTag;

  Result.flush();
}

void CommentASTToXMLConverter::appendToResultWithXMLEscaping(StringRef S) {
  for (StringRef::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    const char C = *I;
    switch (C) {
      case '&':
        Result << "&amp;";
        break;
      case '<':
        Result << "&lt;";
        break;
      case '>':
        Result << "&gt;";
        break;
      case '"':
        Result << "&quot;";
        break;
      case '\'':
        Result << "&apos;";
        break;
      default:
        Result << C;
        break;
    }
  }
}

extern "C" {

CXString clang_FullComment_getAsXML(CXComment CXC) {
  const FullComment *FC = getASTNodeAs<FullComment>(CXC);
  if (!FC)
    return createCXString((const char *) 0);

  CXTranslationUnit TU = CXC.TranslationUnit;
  SourceManager &SM = static_cast<ASTUnit *>(TU->TUData)->getSourceManager();

  SmallString<1024> XML;
  CommentASTToXMLConverter Converter(FC, XML, getCommandTraits(CXC), SM);
  Converter.visit(FC);
  return createCXString(XML.str(), /* DupString = */ true);
}

} // end extern "C"

