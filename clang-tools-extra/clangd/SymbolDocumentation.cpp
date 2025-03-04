//===--- SymbolDocumentation.cpp ==-------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolDocumentation.h"
#include "Config.h"
#include "support/Markup.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"

namespace clang {
namespace clangd {

void commandToMarkup(markup::Paragraph &Out, StringRef Command,
                     comments::CommandMarkerKind CommandMarker,
                     StringRef Args) {
  Out.appendBoldText(
      (CommandMarker == (comments::CommandMarkerKind::CMK_At) ? "@" : "\\") +
      Command.str());
  if (!Args.empty()) {
    Out.appendSpace();
    Out.appendEmphasizedText(Args.str());
  }
}

class ParagraphToMarkupDocument
    : public comments::ConstCommentVisitor<ParagraphToMarkupDocument> {
public:
  ParagraphToMarkupDocument(markup::Paragraph &Out,
                            const comments::CommandTraits &Traits)
      : Out(Out), Traits(Traits) {}

  void visitParagraphComment(const comments::ParagraphComment *C) {
    for (const auto *Child = C->child_begin(); Child != C->child_end();
         ++Child) {
      visit(*Child);
    }
  }

  void visitTextComment(const comments::TextComment *C) {
    Out.appendText(C->getText().str());
    // A paragraph may have multiple TextComments seperated by a newline.
    // We need to add a space to separate them in the markup::Document.
    if (C->hasTrailingNewline())
      Out.appendSpace();
  }

  void visitInlineCommandComment(const comments::InlineCommandComment *C) {

    if (C->getNumArgs() > 0) {
      std::string ArgText;
      for (unsigned I = 0; I < C->getNumArgs(); ++I) {
        if (!ArgText.empty())
          ArgText += " ";
        ArgText += C->getArgText(I);
      }

      switch (C->getRenderKind()) {
      case comments::InlineCommandRenderKind::Monospaced:
        Out.appendCode(ArgText);
        break;
      case comments::InlineCommandRenderKind::Bold:
        Out.appendBoldText(ArgText);
        break;
      case comments::InlineCommandRenderKind::Emphasized:
        Out.appendEmphasizedText(ArgText);
        break;
      default:
        commandToMarkup(Out, C->getCommandName(Traits), C->getCommandMarker(),
                        ArgText);
        break;
      }
    } else {
      commandToMarkup(Out, C->getCommandName(Traits), C->getCommandMarker(),
                      "");
    }
  }

private:
  markup::Paragraph &Out;
  const comments::CommandTraits &Traits;
};

class BlockCommentToMarkupDocument
    : public comments::ConstCommentVisitor<BlockCommentToMarkupDocument> {
public:
  BlockCommentToMarkupDocument(markup::Document &Out,
                               const comments::CommandTraits &Traits)
      : Out(Out), Traits(Traits) {}

  void visitBlockCommandComment(const comments::BlockCommandComment *B) {
    // Some commands have arguments, like \throws.
    // The arguments are not part of the paragraph.
    // We need reconstruct them here.
    std::string ArgText;
    for (unsigned I = 0; I < B->getNumArgs(); ++I) {
      if (!ArgText.empty())
        ArgText += " ";
      ArgText += B->getArgText(I);
    }

    auto &P = Out.addParagraph();

    commandToMarkup(P, B->getCommandName(Traits), B->getCommandMarker(),
                    ArgText);

    if (!B->getParagraph()->isWhitespace()) {
      P.appendSpace();
      ParagraphToMarkupDocument(P, Traits).visit(B->getParagraph());
    }
  }

private:
  markup::Document &Out;
  const comments::CommandTraits &Traits;
};

class FullCommentToMarkupDocument
    : public comments::ConstCommentVisitor<FullCommentToMarkupDocument> {
public:
  FullCommentToMarkupDocument(
      const comments::FullComment &FC, const comments::CommandTraits &Traits,
      markup::Document &Doc, std::optional<SymbolPrintedType> SymbolType,
      std::optional<SymbolPrintedType> SymbolReturnType,
      const std::optional<std::vector<SymbolParam>> &SymbolParameters)
      : Traits(Traits), Output(Doc) {

    for (auto *Block : FC.getBlocks()) {
      visit(Block);
    }

    for (const auto *BP : BriefParagraphs)
      ParagraphToMarkupDocument(Output.addParagraph(), Traits).visit(BP);

    if (!BriefParagraphs.empty())
      Output.addRuler();

    if (SymbolReturnType.has_value()) {
      std::string RT = llvm::to_string(*SymbolReturnType);
      if (!RT.empty()) {
        auto &P = Output.addParagraph().appendText("→ ").appendCode(RT);
        if (!ReturnParagraphs.empty()) {
          P.appendText(": ");
          for (const auto *RP : ReturnParagraphs)
            ParagraphToMarkupDocument(P, Traits).visit(RP);
        }
      }
    }

    if (SymbolParameters.has_value() && !SymbolParameters->empty()) {
      Output.addParagraph().appendText("Parameters:");
      markup::BulletList &L = Output.addBulletList();
      for (const auto &P : *SymbolParameters) {
        markup::Paragraph &PP =
            L.addItem().addParagraph().appendCode(llvm::to_string(P));

        if (!P.Name.has_value())
          continue;

        if (const auto *PCC = Parameters[*P.Name]) {
          PP.appendText(": ");
          ParagraphToMarkupDocument(PP, Traits).visit(PCC->getParagraph());
          Parameters.erase(*P.Name);
        }
      }
    }

    // Don't print Type after Parameters or ReturnType as this will just
    // duplicate the information
    if (SymbolType.has_value() && !SymbolReturnType.has_value() &&
        (!SymbolParameters.has_value() || SymbolParameters->empty())) {
      Output.addParagraph().appendText("Type: ").appendCode(
          llvm::to_string(*SymbolType));
    }

    if (!WarningParagraphs.empty()) {
      Output.addParagraph().appendText("Warning").appendText(
          WarningParagraphs.size() > 1 ? "s:" : ":");
      markup::BulletList &L = Output.addBulletList();
      for (const auto *WP : WarningParagraphs)
        ParagraphToMarkupDocument(L.addItem().addParagraph(), Traits).visit(WP);
      Output.addRuler();
    }

    if (!NoteParagraphs.empty()) {
      if (WarningParagraphs.empty())
        Output.addRuler();
      Output.addParagraph().appendText("Note").appendText(
          WarningParagraphs.size() > 1 ? "s:" : ":");
      markup::BulletList &L = Output.addBulletList();
      for (const auto *WP : NoteParagraphs)
        ParagraphToMarkupDocument(L.addItem().addParagraph(), Traits).visit(WP);
      Output.addRuler();
    }

    for (unsigned I = 0; I < CommentPartIndex; ++I) {
      if (const auto *UnhandledCommand = UnhandledCommands.lookup(I)) {
        BlockCommentToMarkupDocument(Output, Traits).visit(UnhandledCommand);
        continue;
      }
      if (const auto *FreeText = FreeParagraphs.lookup(I)) {
        ParagraphToMarkupDocument(Output.addParagraph(), Traits)
            .visit(FreeText);
        continue;
      }
    }
  }

  void visitBlockCommandComment(const comments::BlockCommandComment *B) {
    const comments::CommandInfo *Info =
        Traits.getCommandInfo(B->getCommandID());

    // Visit B->getParagraph() for commands that we have special fields for,
    // so that the command name won't be included in the string.
    // Otherwise, we want to keep the command name, so visit B itself.
    if (Info->IsBriefCommand) {
      BriefParagraphs.push_back(B->getParagraph());
    } else if (Info->IsReturnsCommand) {
      ReturnParagraphs.push_back(B->getParagraph());
    } else {
      const llvm::StringRef CommandName = B->getCommandName(Traits);
      if (CommandName == "warning") {
        WarningParagraphs.push_back(B->getParagraph());
      } else if (CommandName == "note") {
        NoteParagraphs.push_back(B->getParagraph());
      } else {
        UnhandledCommands[CommentPartIndex] = B;
      }
      CommentPartIndex++;
    }
  }

  void visitParagraphComment(const comments::ParagraphComment *P) {
    FreeParagraphs[CommentPartIndex] = P;
    CommentPartIndex++;
  }

  void visitParamCommandComment(const comments::ParamCommandComment *P) {
    Parameters[P->getParamNameAsWritten()] = P;
  }

private:
  const comments::CommandTraits &Traits;
  markup::Document &Output;

  unsigned CommentPartIndex = 0;

  /// Paragraph of the "brief" command.
  llvm::SmallVector<const comments::ParagraphComment *, 1> BriefParagraphs;

  /// Paragraph of the "return" command.
  llvm::SmallVector<const comments::ParagraphComment *, 1> ReturnParagraphs;

  /// Paragraph(s) of the "note" command(s)
  llvm::SmallVector<const comments::ParagraphComment *> NoteParagraphs;

  /// Paragraph(s) of the "warning" command(s)
  llvm::SmallVector<const comments::ParagraphComment *> WarningParagraphs;

  /// Parsed paragaph(s) of the "param" comamnd(s)
  llvm::SmallDenseMap<StringRef, const comments::ParamCommandComment *>
      Parameters;

  /// All the paragraphs we don't have any special handling for,
  /// e.g. "details".
  llvm::SmallDenseMap<unsigned, const comments::BlockCommandComment *>
      UnhandledCommands;

  /// All free text paragraphs.
  llvm::SmallDenseMap<unsigned, const comments::ParagraphComment *>
      FreeParagraphs;
};

void fullCommentToMarkupDocument(
    markup::Document &Doc, const comments::FullComment *FC,
    const comments::CommandTraits &Traits,
    std::optional<SymbolPrintedType> SymbolType,
    std::optional<SymbolPrintedType> SymbolReturnType,
    const std::optional<std::vector<SymbolParam>> &SymbolParameters) {
  if (!FC)
    return;
  FullCommentToMarkupDocument(*FC, Traits, Doc, SymbolType, SymbolReturnType,
                              SymbolParameters);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SymbolPrintedType &T) {
  OS << T.Type;
  if (T.AKA)
    OS << " (aka " << *T.AKA << ")";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolParam &P) {
  if (P.Type)
    OS << P.Type->Type;
  if (P.Name)
    OS << " " << *P.Name;
  if (P.Default)
    OS << " = " << *P.Default;
  if (P.Type && P.Type->AKA)
    OS << " (aka " << *P.Type->AKA << ")";
  return OS;
}

SymbolPrintedType printType(QualType QT, ASTContext &ASTCtx,
                            const PrintingPolicy &PP) {
  // TypePrinter doesn't resolve decltypes, so resolve them here.
  // FIXME: This doesn't handle composite types that contain a decltype in them.
  // We should rather have a printing policy for that.
  while (!QT.isNull() && QT->isDecltypeType())
    QT = QT->castAs<DecltypeType>()->getUnderlyingType();
  SymbolPrintedType Result;
  llvm::raw_string_ostream OS(Result.Type);
  // Special case: if the outer type is a tag type without qualifiers, then
  // include the tag for extra clarity.
  // This isn't very idiomatic, so don't attempt it for complex cases, including
  // pointers/references, template specializations, etc.
  if (!QT.isNull() && !QT.hasQualifiers() && PP.SuppressTagKeyword) {
    if (auto *TT = llvm::dyn_cast<TagType>(QT.getTypePtr()))
      OS << TT->getDecl()->getKindName() << " ";
  }
  QT.print(OS, PP);

  const Config &Cfg = Config::current();
  if (!QT.isNull() && Cfg.Hover.ShowAKA) {
    bool ShouldAKA = false;
    QualType DesugaredTy = clang::desugarForDiagnostic(ASTCtx, QT, ShouldAKA);
    if (ShouldAKA)
      Result.AKA = DesugaredTy.getAsString(PP);
  }
  return Result;
}

SymbolPrintedType printType(const TemplateTypeParmDecl *TTP) {
  SymbolPrintedType Result;
  Result.Type = TTP->wasDeclaredWithTypename() ? "typename" : "class";
  if (TTP->isParameterPack())
    Result.Type += "...";
  return Result;
}

SymbolPrintedType printType(const NonTypeTemplateParmDecl *NTTP,
                            const PrintingPolicy &PP) {
  auto PrintedType = printType(NTTP->getType(), NTTP->getASTContext(), PP);
  if (NTTP->isParameterPack()) {
    PrintedType.Type += "...";
    if (PrintedType.AKA)
      *PrintedType.AKA += "...";
  }
  return PrintedType;
}

SymbolPrintedType printType(const TemplateTemplateParmDecl *TTP,
                            const PrintingPolicy &PP) {
  SymbolPrintedType Result;
  llvm::raw_string_ostream OS(Result.Type);
  OS << "template <";
  llvm::StringRef Sep = "";
  for (const Decl *Param : *TTP->getTemplateParameters()) {
    OS << Sep;
    Sep = ", ";
    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
      OS << printType(TTP).Type;
    else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param))
      OS << printType(NTTP, PP).Type;
    else if (const auto *TTPD = dyn_cast<TemplateTemplateParmDecl>(Param))
      OS << printType(TTPD, PP).Type;
  }
  // FIXME: TemplateTemplateParameter doesn't store the info on whether this
  // param was a "typename" or "class".
  OS << "> class";
  return Result;
}

// Default argument might exist but be unavailable, in the case of unparsed
// arguments for example. This function returns the default argument if it is
// available.
const Expr *getDefaultArg(const ParmVarDecl *PVD) {
  // Default argument can be unparsed or uninstantiated. For the former we
  // can't do much, as token information is only stored in Sema and not
  // attached to the AST node. For the latter though, it is safe to proceed as
  // the expression is still valid.
  if (!PVD->hasDefaultArg() || PVD->hasUnparsedDefaultArg())
    return nullptr;
  return PVD->hasUninstantiatedDefaultArg() ? PVD->getUninstantiatedDefaultArg()
                                            : PVD->getDefaultArg();
}

SymbolParam createSymbolParam(const ParmVarDecl *PVD,
                              const PrintingPolicy &PP) {
  SymbolParam Out;
  Out.Type = printType(PVD->getType(), PVD->getASTContext(), PP);
  if (!PVD->getName().empty())
    Out.Name = PVD->getNameAsString();
  if (const Expr *DefArg = getDefaultArg(PVD)) {
    Out.Default.emplace();
    llvm::raw_string_ostream OS(*Out.Default);
    DefArg->printPretty(OS, nullptr, PP);
  }
  return Out;
}

} // namespace clangd
} // namespace clang
