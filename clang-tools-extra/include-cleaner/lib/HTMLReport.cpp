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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::include_cleaner {
namespace {

constexpr llvm::StringLiteral CSS = R"css(
  body { margin: 0; }
  pre { line-height: 1.5em; counter-reset: line; margin: 0; }
  pre .line:not(.added) { counter-increment: line; }
  pre .line::before {
    content: counter(line);
    display: inline-block;
    background-color: #eee; border-right: 1px solid #ccc;
    text-align: right;
    width: 3em; padding-right: 0.5em; margin-right: 0.5em;
  }
  pre .line.added::before { content: '+' }
  .ref, .inc { text-decoration: underline; color: #008; }
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
  #hover .target.implicit, .provides .implicit { background-color: #bbb; }
  #hover .target.ambiguous, .provides .ambiguous { background-color: #caf; }
  .missing, .unused { background-color: #faa !important; }
  .inserted { background-color: #bea !important; }
  .semiused { background-color: #888 !important; }
  #hover th { color: #008; text-align: right; padding-right: 0.5em; }
  #hover .target:not(:first-child) {
    margin-top: 1em;
    padding-top: 1em;
    border-top: 1px solid #444;
  }
  .ref.missing #hover .insert { font-weight: bold; }
  .ref:not(.missing) #hover .insert { font-style: italic; }
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

// Return detailed symbol description (declaration), if we have any.
std::string printDetails(const Symbol &Sym) {
  std::string S;
  if (Sym.kind() == Symbol::Declaration) {
    // Print the declaration of the symbol, e.g. to disambiguate overloads.
    const auto &D = Sym.declaration();
    PrintingPolicy PP = D.getASTContext().getPrintingPolicy();
    PP.FullyQualifiedName = true;
    PP.TerseOutput = true;
    PP.SuppressInitializers = true;
    llvm::raw_string_ostream SS(S);
    D.print(SS, PP);
  }
  return S;
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
  HeaderSearch &HS;
  const RecordedPP::RecordedIncludes &Includes;
  const PragmaIncludes *PI;
  FileID MainFile;
  const FileEntry *MainFE;

  // References to symbols from the main file.
  // FIXME: should we deduplicate these?
  struct Target {
    Symbol Sym;
    RefType Type;
    SmallVector<SymbolLocation> Locations;
    SmallVector<Header> Headers;
    SmallVector<const Include *> Includes;
    bool Satisfied = false; // Is the include present?
    std::string Insert;     // If we had no includes, what would we insert?
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
  llvm::DenseMap<const Include *, std::vector<unsigned>> IncludeRefs;
  llvm::StringMap<std::vector</*RefIndex*/unsigned>> Insertion;

  llvm::StringRef includeType(const Include *I) {
    auto &List = IncludeRefs[I];
    if (List.empty())
      return "unused";
    if (llvm::any_of(List, [&](unsigned I) {
          return Targets[Refs[I].TargetIndex].Type == RefType::Explicit;
        }))
      return "used";
    return "semiused";
  }

  std::string spellHeader(const Header &H) {
    switch (H.kind()) {
    case Header::Physical: {
      bool IsSystem = false;
      std::string Path = HS.suggestPathToFileForDiagnostics(
          H.physical(), MainFE->tryGetRealPathName(), &IsSystem);
      return IsSystem ? "<" + Path + ">" : "\"" + Path + "\"";
    }
    case Header::Standard:
      return H.standard().name().str();
    case Header::Verbatim:
      return H.verbatim().str();
    }
    llvm_unreachable("Unknown Header kind");
  }

  Target makeTarget(const SymbolReference &SR) {
    Target T{SR.Target, SR.RT, {}, {}, {}, {}, {}};

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
      T.Headers.append(findHeaders(Loc, SM, PI));

    for (const auto &H : T.Headers) {
      T.Includes.append(Includes.match(H));
      // FIXME: library should signal main-file refs somehow.
      // Non-physical refs to the main-file should be possible.
      if (H.kind() == Header::Physical && H.physical() == MainFE)
        T.Satisfied = true;
    }
    if (!T.Includes.empty())
      T.Satisfied = true;
    // Include pointers are meaningfully ordered as they are backed by a vector.
    llvm::sort(T.Includes);
    T.Includes.erase(std::unique(T.Includes.begin(), T.Includes.end()),
                     T.Includes.end());
    
    if (!T.Headers.empty())
      // FIXME: library should tell us which header to use.
      T.Insert = spellHeader(T.Headers.front());

    return T;
  }

public:
  Reporter(llvm::raw_ostream &OS, ASTContext &Ctx, HeaderSearch &HS,
           const RecordedPP::RecordedIncludes &Includes,
           const PragmaIncludes *PI, FileID MainFile)
      : OS(OS), Ctx(Ctx), SM(Ctx.getSourceManager()), HS(HS),
        Includes(Includes), PI(PI), MainFile(MainFile),
        MainFE(SM.getFileEntryForID(MainFile)) {}

  void addRef(const SymbolReference &SR) {
    auto [File, Offset] = SM.getDecomposedLoc(SM.getFileLoc(SR.RefLocation));
    if (File != this->MainFile) {
      // Can get here e.g. if there's an #include inside a root Decl.
      // FIXME: do something more useful than this.
      llvm::errs() << "Ref location outside file! " << SR.Target << " at "
                   << SR.RefLocation.printToString(SM) << "\n";
      return;
    }

    Refs.push_back({Offset, SR.RT == RefType::Implicit, Targets.size()});
    Targets.push_back(makeTarget(SR));
    for (const auto *I : Targets.back().Includes)
      IncludeRefs[I].push_back(Targets.size() - 1);
    if (Targets.back().Type == RefType::Explicit && !Targets.back().Satisfied &&
        !Targets.back().Insert.empty())
      Insertion[Targets.back().Insert].push_back(Targets.size() - 1);
  }

  void write() {
    OS << "<!doctype html>\n";
    OS << "<html>\n";
    OS << "<head>\n";
    OS << "<style>" << CSS << "</style>\n";
    OS << "<script>" << JS << "</script>\n";
    for (const auto &Ins : Insertion) {
      OS << "<template id='i";
      escapeString(Ins.first());
      OS << "'>";
      writeInsertion(Ins.first(), Ins.second);
      OS << "</template>\n";
    }
    for (auto &Inc : Includes.all()) {
      OS << "<template id='i" << Inc.Line << "'>";
      writeInclude(Inc);
      OS << "</template>\n";
    }
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
  
  // Write "Provides: " rows of an include or include-insertion table.
  // These describe the symbols the header provides, referenced by RefIndices.
  void writeProvides(llvm::ArrayRef<unsigned> RefIndices) {
    // We show one ref for each symbol: first by (RefType != Explicit, Sequence)
    llvm::DenseMap<Symbol, /*RefIndex*/ unsigned> FirstRef;
    for (unsigned RefIndex : RefIndices) {
      const Target &T = Targets[Refs[RefIndex].TargetIndex];
      auto I = FirstRef.try_emplace(T.Sym, RefIndex);
      if (!I.second && T.Type == RefType::Explicit &&
          Targets[Refs[I.first->second].TargetIndex].Type != RefType::Explicit)
        I.first->second = RefIndex;
    }
    std::vector<std::pair<Symbol, unsigned>> Sorted = {FirstRef.begin(),
                                                       FirstRef.end()};
    llvm::stable_sort(Sorted, llvm::less_second{});
    for (auto &[S, RefIndex] : Sorted) {
      auto &T = Targets[Refs[RefIndex].TargetIndex];
      OS << "<tr class='provides'><th>Provides</td><td>";
      std::string Details = printDetails(S);
      if (!Details.empty()) {
        OS << "<span class='" << refType(T.Type) << "' title='";
        escapeString(Details);
        OS << "'>";
      }
      escapeString(llvm::to_string(S));
      if (!Details.empty())
        OS << "</span>";
      
      unsigned Line = SM.getLineNumber(MainFile, Refs[RefIndex].Offset);
      OS << ", <a href='#line" << Line << "'>line " << Line << "</a>";
      OS << "</td></tr>";
    }
  }

  void writeInclude(const Include &Inc) {
    OS << "<table class='include'>";
    if (Inc.Resolved) {
      OS << "<tr><th>Resolved</td><td>";
      escapeString(Inc.Resolved->getName());
      OS << "</td></tr>\n";
      writeProvides(IncludeRefs[&Inc]);
    }
    OS << "</table>";
  }
  
  void writeInsertion(llvm::StringRef Text, llvm::ArrayRef<unsigned> Refs) {
    OS << "<table class='insertion'>";
    writeProvides(Refs);
    OS << "</table>";
  }

  void writeTarget(const Target &T) {
    OS << "<table class='target " << refType(T.Type) << "'>";

    OS << "<tr><th>Symbol</th><td>";
    OS << describeSymbol(T.Sym) << " <code>";
    escapeString(llvm::to_string(T.Sym));
    OS << "</code></td></tr>\n";

    std::string Details = printDetails(T.Sym);
    if (!Details.empty()) {
      OS << "<tr><td></td><td><code>";
      escapeString(Details);
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

    for (const auto *I : T.Includes) {
      OS << "<tr><th>Included</th><td>";
      escapeString(I->Spelled);
      OS << ", <a href='#line" << I->Line << "'>line " << I->Line << "</a>";
      OS << "</td></tr>";
    }
    
    if (!T.Insert.empty()) {
      OS << "<tr><th>Insert</th><td class='insert'>";
      escapeString(T.Insert);
      OS << "</td></tr>";
    }

    OS << "</table>";
  }

  void writeCode() {
    llvm::sort(Refs);
    llvm::StringRef Code = SM.getBufferData(MainFile);

    OS << "<pre onclick='select(event)' class='code'>";

    std::vector<llvm::StringRef> Insertions{Insertion.keys().begin(),
                                            Insertion.keys().end()};
    llvm::sort(Insertions);
    for (llvm::StringRef Insertion : Insertions) {
      OS << "<code class='line added'>"
          << "<span class='inc sel inserted' data-hover='i";
      escapeString(Insertion);
      OS << "'>#include ";
      escapeString(Insertion);
      OS << "</span></code>\n";
    }

    const Include *Inc = nullptr;
    unsigned LineNum = 0;
    // Lines are <code>, include lines have an inner <span>.
    auto StartLine = [&] {
      ++LineNum;
      OS << "<code class='line' id='line" << LineNum << "'>";
      if ((Inc = Includes.atLine(LineNum)))
        OS << "<span class='inc sel " << includeType(Inc) << "' data-hover='i"
           << Inc->Line << "'>";
    };
    auto EndLine = [&] {
      if (Inc)
        OS << "</span>";
      OS << "</code>\n";
    };

    auto Rest = llvm::makeArrayRef(Refs);
    unsigned End = 0;
    StartLine();
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
        OS << "<span class='ref sel implicit "
           << (Targets[R.TargetIndex].Satisfied ? "satisfied" : "missing")
           << "' data-hover='t" << R.TargetIndex << "'>&loz;</span>";
        Rest = Rest.drop_front();
      };
      // Accumulate all explicit refs that appear on the same token.
      std::string TargetList;
      bool Unsatisfied = false;
      Rest = Rest.drop_while([&](const Ref &R) {
        if (R.Offset != I)
          return false;
        if (!TargetList.empty())
          TargetList.push_back(',');
        TargetList.push_back('t');
        TargetList.append(std::to_string(R.TargetIndex));
        Unsatisfied = Unsatisfied || !Targets[R.TargetIndex].Satisfied;
        return true;
      });
      if (!TargetList.empty()) {
        assert(End == 0 && "Overlapping tokens!");
        OS << "<span class='ref sel" << (Unsatisfied ? " missing" : "")
           << "' data-hover='" << TargetList << "'>";
        End = I + Lexer::MeasureTokenLength(SM.getComposedLoc(MainFile, I), SM,
                                            Ctx.getLangOpts());
      }
      if (Code[I] == '\n') {
        EndLine();
        StartLine();
      } else
        escapeChar(Code[I]);
    }
    EndLine();
    OS << "</pre>\n";
  }
};

} // namespace

void writeHTMLReport(FileID File, const RecordedPP::RecordedIncludes &Includes,
                     llvm::ArrayRef<Decl *> Roots,
                     llvm::ArrayRef<SymbolReference> MacroRefs, ASTContext &Ctx,
                     HeaderSearch &HS, PragmaIncludes *PI,
                     llvm::raw_ostream &OS) {
  Reporter R(OS, Ctx, HS, Includes, PI, File);
  for (Decl *Root : Roots)
    walkAST(*Root, [&](SourceLocation Loc, const NamedDecl &D, RefType T) {
      R.addRef(SymbolReference{Loc, D, T});
    });
  for (const SymbolReference &Ref : MacroRefs)
    R.addRef(Ref);
  R.write();
}

} // namespace clang::include_cleaner
