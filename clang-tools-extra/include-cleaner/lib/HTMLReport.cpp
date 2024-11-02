//===--- HTMLReport.cpp - Explain the analysis for humans -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// If we're debugging this tool or trying to explain its conclusions, we need to
// be able to identify specific facts about the code and the inferences made.
//
// This library prints an annotated version of the code
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::include_cleaner {
namespace {

constexpr llvm::StringLiteral CSS = R"css(
  body { margin: 0; }
  pre { line-height: 1.5em; counter-reset: line; margin: 0; }
  pre .line { counter-increment: line; }
  pre .line::before {
    content: counter(line);
    display: inline-block;
    background-color: #eee; border-right: 1px solid #ccc;
    text-align: right;
    width: 3em; padding-right: 0.5em; margin-right: 0.5em;
  }
  .ref { text-decoration: underline; color: #008; }
  .sel { position: relative; cursor: pointer; }
  #hover {
    background-color: #aaccff; border: 1px solid black;
    z-index: 1;
    position: absolute; top: 100%; left: 0;
    font-family: sans-serif;
    padding: 0.5em;
  }
  #hover p, #hover pre { margin: 0; }
  #hover section header { font-weight: bold; }
  #hover section:not(:first-child) { margin-top: 1em; }
)css";

constexpr llvm::StringLiteral JS = R"js(
  // Recreate the #hover div inside whichever target .sel element was clicked.
  function select(event) {
    var target = event.target.closest('.sel');
    var hover = document.getElementById('hover');
    if (hover) {
      if (hover.parentElement == target) return;
      hover.parentNode.removeChild(hover);
    }
    if (target == null) return;
    hover = document.createElement('div');
    hover.id = 'hover';
    fillHover(hover, target);
    target.appendChild(hover);
  }
  // Fill the #hover div with the templates named by data-hover in the target.
  function fillHover(hover, target) {
    target.dataset.hover?.split(',').forEach(function(id) {
      for (c of document.getElementById(id).content.childNodes)
        hover.appendChild(c.cloneNode(true));
    })
  }
)js";

// Print the declaration tersely, but enough to identify e.g. which overload.
std::string printDecl(const NamedDecl &ND) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  PrintingPolicy PP = ND.getASTContext().getPrintingPolicy();
  PP.FullyQualifiedName = true;
  PP.TerseOutput = true;
  PP.SuppressInitializers = true;
  ND.print(OS, PP);
  llvm::erase_value(S, '\n');
  return S;
}

class Reporter {
  llvm::raw_ostream &OS;
  const ASTContext &Ctx;
  const SourceManager &SM;
  FileID File;

  // Symbols that are referenced from the main file.
  struct Target {
    const NamedDecl *D;
  };
  std::vector<Target> Targets;
  // Points within the main file that reference a Target.
  std::vector<std::pair</*Offset*/ unsigned, /*TargetIndex*/ unsigned>> Refs;

public:
  Reporter(llvm::raw_ostream &OS, ASTContext &Ctx, FileID File)
      : OS(OS), Ctx(Ctx), SM(Ctx.getSourceManager()), File(File) {}

  void addRef(SourceLocation Loc, const NamedDecl &D) {
    auto [File, Offset] = SM.getDecomposedLoc(SM.getFileLoc(Loc));
    if (File != this->File) {
      // Can get here e.g. if there's an #include inside a root Decl.
      // FIXME: do something more useful than this.
      llvm::errs() << "Ref location outside file! "
                   << D.getQualifiedNameAsString() << " at "
                   << Loc.printToString(SM) << "\n";
      return;
    }
    Targets.push_back({&D});
    Refs.push_back({Offset, Targets.size() - 1});
  }

  void write() {
    OS << "<!doctype html>\n";
    OS << "<html>\n";
    OS << "<head>\n";
    OS << "<style>" << CSS << "</style>\n";
    OS << "<script>" << JS << "</script>\n";
    for (unsigned I = 0; I < Targets.size(); ++I) {
      OS << "<template id='t" << I << "'><section>";
      writeTarget(Targets[I]);
      OS << "</section></template>\n";
    }
    OS << "</head>\n";
    OS << "<body>\n";
    writeCode();
    OS << "</body>\n";
    OS << "</html>\n";
  }

private:
  void escapeChar(char C) {
    switch (C) {
    case '<':
      OS << "&lt;";
      break;
    case '&':
      OS << "&amp;";
      break;
    default:
      OS << C;
    }
  }

  void escapeString(llvm::StringRef S) {
    for (char C : S)
      escapeChar(C);
  }

  void writeTarget(const Target &T) {
    OS << "<header>" << T.D->getDeclKindName() << " ";
    escapeString(T.D->getQualifiedNameAsString());
    OS << "</header>";

    OS << "<p>declared at ";
    escapeString(SM.getFileLoc(T.D->getLocation()).printToString(SM));
    OS << "</p><pre>";
    escapeString(printDecl(*T.D));
    OS << "</pre>";
  }

  void writeCode() {
    llvm::sort(Refs);
    llvm::StringRef Code = SM.getBufferData(File);

    OS << "<pre onclick='select(event)' class='code'>";
    OS << "<code class='line'>";
    auto Rest = llvm::makeArrayRef(Refs);
    unsigned End = 0;
    for (unsigned I = 0; I < Code.size(); ++I) {
      // Finish refs early at EOL to avoid dealing with splitting the span.
      if (End && (End == I || Code[I] == '\n')) {
        OS << "</span>";
        End = 0;
      }
      std::string TargetList;
      Rest = Rest.drop_while([&](auto &R) {
        if (R.first != I)
          return false;
        if (!TargetList.empty())
          TargetList.push_back(',');
        TargetList.push_back('t');
        TargetList.append(std::to_string(R.second));
        return true;
      });
      if (!TargetList.empty()) {
        assert(End == 0 && "Overlapping tokens!");
        OS << "<span class='ref sel' data-hover='" << TargetList << "'>";
        End = I + Lexer::MeasureTokenLength(SM.getComposedLoc(File, I), SM,
                                            Ctx.getLangOpts());
      }
      if (Code[I] == '\n')
        OS << "</code>\n<code class='line'>";
      else
        escapeChar(Code[I]);
    }
    OS << "</code></pre>\n";
  }
};

} // namespace

void writeHTMLReport(FileID File, llvm::ArrayRef<Decl *> Roots, ASTContext &Ctx,
                     llvm::raw_ostream &OS) {
  Reporter R(OS, Ctx, File);
  for (Decl *Root : Roots)
    walkAST(*Root, [&](SourceLocation Loc, const NamedDecl &D, RefType) {
      R.addRef(Loc, D);
    });
  R.write();
}

} // namespace clang::include_cleaner
