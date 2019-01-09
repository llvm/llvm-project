//===--- TextNodeDumper.h - Printing of AST nodes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AST dumping of components of individual AST nodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEXTNODEDUMPER_H
#define LLVM_CLANG_AST_TEXTNODEDUMPER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/ExprCXX.h"

namespace clang {

class TextTreeStructure {
  raw_ostream &OS;
  const bool ShowColors;

  /// Pending[i] is an action to dump an entity at level i.
  llvm::SmallVector<std::function<void(bool isLastChild)>, 32> Pending;

  /// Indicates whether we're at the top level.
  bool TopLevel = true;

  /// Indicates if we're handling the first child after entering a new depth.
  bool FirstChild = true;

  /// Prefix for currently-being-dumped entity.
  std::string Prefix;

public:
  /// Add a child of the current node.  Calls doAddChild without arguments
  template <typename Fn> void addChild(Fn doAddChild) {
    // If we're at the top level, there's nothing interesting to do; just
    // run the dumper.
    if (TopLevel) {
      TopLevel = false;
      doAddChild();
      while (!Pending.empty()) {
        Pending.back()(true);
        Pending.pop_back();
      }
      Prefix.clear();
      OS << "\n";
      TopLevel = true;
      return;
    }

    auto dumpWithIndent = [this, doAddChild](bool isLastChild) {
      // Print out the appropriate tree structure and work out the prefix for
      // children of this node. For instance:
      //
      //   A        Prefix = ""
      //   |-B      Prefix = "| "
      //   | `-C    Prefix = "|   "
      //   `-D      Prefix = "  "
      //     |-E    Prefix = "  | "
      //     `-F    Prefix = "    "
      //   G        Prefix = ""
      //
      // Note that the first level gets no prefix.
      {
        OS << '\n';
        ColorScope Color(OS, ShowColors, IndentColor);
        OS << Prefix << (isLastChild ? '`' : '|') << '-';
        this->Prefix.push_back(isLastChild ? ' ' : '|');
        this->Prefix.push_back(' ');
      }

      FirstChild = true;
      unsigned Depth = Pending.size();

      doAddChild();

      // If any children are left, they're the last at their nesting level.
      // Dump those ones out now.
      while (Depth < Pending.size()) {
        Pending.back()(true);
        this->Pending.pop_back();
      }

      // Restore the old prefix.
      this->Prefix.resize(Prefix.size() - 2);
    };

    if (FirstChild) {
      Pending.push_back(std::move(dumpWithIndent));
    } else {
      Pending.back()(false);
      Pending.back() = std::move(dumpWithIndent);
    }
    FirstChild = false;
  }

  TextTreeStructure(raw_ostream &OS, bool ShowColors)
      : OS(OS), ShowColors(ShowColors) {}
};

class TextNodeDumper
    : public TextTreeStructure,
      public comments::ConstCommentVisitor<TextNodeDumper, void,
                                           const comments::FullComment *> {
  raw_ostream &OS;
  const bool ShowColors;

  /// Keep track of the last location we print out so that we can
  /// print out deltas from then on out.
  const char *LastLocFilename = "";
  unsigned LastLocLine = ~0U;

  const SourceManager *SM;

  /// The policy to use for printing; can be defaulted.
  PrintingPolicy PrintPolicy;

  const comments::CommandTraits *Traits;

  const char *getCommandName(unsigned CommandID);

public:
  TextNodeDumper(raw_ostream &OS, bool ShowColors, const SourceManager *SM,
                 const PrintingPolicy &PrintPolicy,
                 const comments::CommandTraits *Traits);

  void Visit(const comments::Comment *C, const comments::FullComment *FC);

  void dumpPointer(const void *Ptr);
  void dumpLocation(SourceLocation Loc);
  void dumpSourceRange(SourceRange R);
  void dumpBareType(QualType T, bool Desugar = true);
  void dumpType(QualType T);
  void dumpBareDeclRef(const Decl *D);
  void dumpName(const NamedDecl *ND);
  void dumpAccessSpecifier(AccessSpecifier AS);
  void dumpCXXTemporary(const CXXTemporary *Temporary);

  void dumpDeclRef(const Decl *D, const char *Label = nullptr);

  void visitTextComment(const comments::TextComment *C,
                        const comments::FullComment *);
  void visitInlineCommandComment(const comments::InlineCommandComment *C,
                                 const comments::FullComment *);
  void visitHTMLStartTagComment(const comments::HTMLStartTagComment *C,
                                const comments::FullComment *);
  void visitHTMLEndTagComment(const comments::HTMLEndTagComment *C,
                              const comments::FullComment *);
  void visitBlockCommandComment(const comments::BlockCommandComment *C,
                                const comments::FullComment *);
  void visitParamCommandComment(const comments::ParamCommandComment *C,
                                const comments::FullComment *FC);
  void visitTParamCommandComment(const comments::TParamCommandComment *C,
                                 const comments::FullComment *FC);
  void visitVerbatimBlockComment(const comments::VerbatimBlockComment *C,
                                 const comments::FullComment *);
  void
  visitVerbatimBlockLineComment(const comments::VerbatimBlockLineComment *C,
                                const comments::FullComment *);
  void visitVerbatimLineComment(const comments::VerbatimLineComment *C,
                                const comments::FullComment *);
};

} // namespace clang

#endif // LLVM_CLANG_AST_TEXTNODEDUMPER_H
