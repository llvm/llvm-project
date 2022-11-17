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
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/Support/ScopedPrinter.h"
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
  .ref.implicit { background-color: #ff8; }
  #hover {
    color: black;
    background-color: #aaccff; border: 1px solid #444;
    z-index: 1;
    position: absolute; top: 100%; left: 0;
    font-family: sans-serif;
    padding: 0.5em;
  }
  #hover p, #hover pre { margin: 0; }
  #hover .target.implicit { background-color: #bbb; }
  #hover .target.ambiguous { background-color: #caf; }
  #hover th { color: #008; text-align: right; padding-right: 0.5em; }
  #hover .target:not(:first-child) {
    margin-top: 1em;
    padding-top: 1em;
    border-top: 1px solid #444;
  }
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

// Categorize the symbol, like FunctionDecl or Macro
llvm::StringRef describeSymbol(const Symbol &Sym) {
  switch (Sym.kind()) {
  case Symbol::Declaration:
    return Sym.declaration().getDeclKindName();
  case Symbol::Macro:
    return "Macro";
  }
  llvm_unreachable("unhandled symbol kind");
}

llvm::StringRef refType(RefType T) {
  switch (T) {
  case RefType::Explicit:
    return "explicit";
  case RefType::Implicit:
    return "implicit";
  case RefType::Ambiguous:
    return "ambiguous";
  }
  llvm_unreachable("unhandled RefType enum");
}

class Reporter {
  llvm::raw_ostream &OS;
  const ASTContext &Ctx;
  const SourceManager &SM;
  const PragmaIncludes *PI;
  FileID File;

  // References to symbols from the main file.
  // FIXME: should we deduplicate these?
  struct Target {
    Symbol Sym;
    RefType Type;
    SmallVector<SymbolLocation> Locations;
    SmallVector<Header> Headers;
  };
  std::vector<Target> Targets;
  // Points within the main file that reference a Target.
  // Implicit refs will be marked with a symbol just before the token.
  struct Ref {
    unsigned Offset;
    bool Implicit;
    size_t TargetIndex;
    bool operator<(const Ref &Other) const {
      return std::forward_as_tuple(Offset, !Implicit, TargetIndex) <
             std::forward_as_tuple(Other.Offset, !Other.Implicit, TargetIndex);
    }
  };
  std::vector<Ref> Refs;

  Target makeTarget(const SymbolReference &SR) {
    Target T{SR.Target, SR.RT, {}, {}};

    // Duplicates logic from walkUsed(), which doesn't expose SymbolLocations.
    // FIXME: use locateDecl and friends once implemented.
    // This doesn't use stdlib::Recognizer, but locateDecl will soon do that.
    switch (SR.Target.kind()) {
    case Symbol::Declaration:
      T.Locations.push_back(SR.Target.declaration().getLocation());
      break;
    case Symbol::Macro:
      T.Locations.push_back(SR.Target.macro().Definition);
      break;
    }

    for (const auto &Loc : T.Locations)
      T.Headers = findHeaders(Loc, SM, PI);

    return T;
  }

public:
  Reporter(llvm::raw_ostream &OS, ASTContext &Ctx, const PragmaIncludes *PI,
           FileID File)
      : OS(OS), Ctx(Ctx), SM(Ctx.getSourceManager()), PI(PI), File(File) {}

  void addRef(const SymbolReference &SR) {
    auto [File, Offset] = SM.getDecomposedLoc(SM.getFileLoc(SR.RefLocation));
    if (File != this->File) {
      // Can get here e.g. if there's an #include inside a root Decl.
      // FIXME: do something more useful than this.
      llvm::errs() << "Ref location outside file! " << SR.Target << " at "
                   << SR.RefLocation.printToString(SM) << "\n";
      return;
    }

    Refs.push_back({Offset, SR.RT == RefType::Implicit, Targets.size()});
    Targets.push_back(makeTarget(SR));
  }

  void write() {
    OS << "<!doctype html>\n";
    OS << "<html>\n";
    OS << "<head>\n";
    OS << "<style>" << CSS << "</style>\n";
    OS << "<script>" << JS << "</script>\n";
    for (unsigned I = 0; I < Targets.size(); ++I) {
      OS << "<template id='t" << I << "'>";
      writeTarget(Targets[I]);
      OS << "</template>\n";
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

  // Abbreviate a path ('path/to/Foo.h') to just the filename ('Foo.h').
  // The full path is available on hover.
  void printFilename(llvm::StringRef Path) {
    llvm::StringRef File = llvm::sys::path::filename(Path);
    if (File == Path)
      return escapeString(Path);
    OS << "<span title='";
    escapeString(Path);
    OS << "'>";
    escapeString(File);
    OS << "</span>";
  }

  // Print a source location in compact style.
  void printSourceLocation(SourceLocation Loc) {
    if (Loc.isInvalid())
      return escapeString("<invalid>");
    if (!Loc.isMacroID())
      return printFilename(Loc.printToString(SM));

    // Replicating printToString() is a bit simpler than parsing/reformatting.
    printFilename(SM.getExpansionLoc(Loc).printToString(SM));
    OS << " &lt;Spelling=";
    printFilename(SM.getSpellingLoc(Loc).printToString(SM));
    OS << ">";
  }

  void writeTarget(const Target &T) {
    OS << "<table class='target " << refType(T.Type) << "'>";

    OS << "<tr><th>Symbol</th><td>";
    OS << describeSymbol(T.Sym) << " <code>";
    escapeString(llvm::to_string(T.Sym));
    OS << "</code></td></tr>\n";

    if (T.Sym.kind() == Symbol::Declaration) {
      // Print the declaration of the symbol, e.g. to disambiguate overloads.
      const auto &D = T.Sym.declaration();
      PrintingPolicy PP = D.getASTContext().getPrintingPolicy();
      PP.FullyQualifiedName = true;
      PP.TerseOutput = true;
      PP.SuppressInitializers = true;
      std::string S;
      llvm::raw_string_ostream SS(S);
      D.print(SS, PP);

      OS << "<tr><td></td><td><code>";
      escapeString(S);
      OS << "</code></td></tr>\n";
    }

    for (const auto &Loc : T.Locations) {
      OS << "<tr><th>Location</th><td>";
      if (Loc.kind() == SymbolLocation::Physical) // needs SM to print properly.
        printSourceLocation(Loc.physical());
      else
        escapeString(llvm::to_string(Loc));
      OS << "</td></tr>\n";
    }

    for (const auto &H : T.Headers) {
      OS << "<tr><th>Header</th><td>";
      switch (H.kind()) {
      case Header::Physical:
        printFilename(H.physical()->getName());
        break;
      case Header::Standard:
        OS << "stdlib " << H.standard().name();
        break;
      case Header::Verbatim:
        OS << "verbatim ";
        escapeString(H.verbatim());
        break;
      }
      OS << "</td></tr>\n";
    }

    OS << "</table>";
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
      // Handle implicit refs, which are rendered *before* the token.
      while (!Rest.empty() && Rest.front().Offset == I &&
             Rest.front().Implicit) {
        const Ref &R = Rest.front();
        OS << "<span class='ref sel implicit' data-hover='t" << R.TargetIndex
           << "'>&loz;</span>";
        Rest = Rest.drop_front();
      };
      // Accumulate all explicit refs that appear on the same token.
      std::string TargetList;
      Rest = Rest.drop_while([&](const Ref &R) {
        if (R.Offset != I)
          return false;
        if (!TargetList.empty())
          TargetList.push_back(',');
        TargetList.push_back('t');
        TargetList.append(std::to_string(R.TargetIndex));
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

void writeHTMLReport(FileID File, llvm::ArrayRef<Decl *> Roots,
                     llvm::ArrayRef<SymbolReference> MacroRefs, ASTContext &Ctx,
                     PragmaIncludes *PI, llvm::raw_ostream &OS) {
  Reporter R(OS, Ctx, PI, File);
  for (Decl *Root : Roots)
    walkAST(*Root, [&](SourceLocation Loc, const NamedDecl &D, RefType T) {
      R.addRef(SymbolReference{Loc, D, T});
    });
  for (const SymbolReference &Ref : MacroRefs)
    R.addRef(Ref);
  R.write();
}

} // namespace clang::include_cleaner
