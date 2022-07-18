//===-- HTMLForest.cpp - browser-based parse forest explorer
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The plain text forest node dump (clang-pseudo -print-forest) is useful but
// hard to reconcile with the code being examined, especially when it is large.
//
// HTMLForest produces a self-contained HTML file containing both the code and
// the forest representation, linking them interactively with javascript.
// At any given time, a single parse tree is shown (ambiguities resolved).
// The user can switch between ambiguous alternatives.
//
// +-------+---------------+
// |       |        +-----+|
// | #tree |  #code |#info||
// |       |        +-----+|
// |       |               |
// +-------+---------------+
//
// #tree is a hierarchical view of the nodes (nested <ul>s), like -print-forest.
// (It is a simple tree, not a DAG, because ambiguities have been resolved).
// Like -print-forest, trivial sequences are collapsed (expression~IDENTIFIER).
//
// #code is the source code, annotated with <span>s marking the node ranges.
// These spans are usually invisible (exception: ambiguities are marked), but
// they are used to show and change the selection.
//
// #info is a floating box that shows details of the currently selected node:
//  - rule (for sequence nodes). Abbreviated rules are also shown.
//  - alternatives (for ambiguous nodes). The user can choose an alternative.
//  - ancestors. The parent nodes show how this node fits in translation-unit.
//
// There are two types of 'active' node:
//  - *highlight* is what the cursor is over, and is colored blue.
//    Near ancestors are shaded faintly (onion-skin) to show local structure.
//  - *selection* is set by clicking.
//    The #info box shows the selection, and selected nodes have a dashed ring.
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Forest.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
namespace clang {
namespace pseudo {
namespace {

// Defines const char HTMLForest_css[] = "...contents of HTMLForest.css..."; etc
#include "HTMLForestResources.inc"

struct Writer {
  llvm::raw_ostream &Out;
  const Grammar &G;
  const ForestNode &Root;
  const TokenStream &Stream;

  void write() {
    Out << "<!doctype html>\n";
    tag("html", [&] {
      tag("head", [&] {
        tag("title", [&] { Out << "HTMLForest"; });
        tag("script", [&] { Out << HTMLForest_js; });
        tag("style", [&] { Out << HTMLForest_css; });
        tag("script", [&] {
          Out << "var forest=";
          writeForestJSON();
          Out << ";";
        });
        tag("pre id='hidden-code' hidden", [&] { writeCode(); });
      });
      tag("body", [&] { Out << HTMLForest_html; });
    });
  }

  void writeCode();
  void writeForestJSON();
  void tag(llvm::StringRef Opener, llvm::function_ref<void()> Body) {
    Out << "<" << Opener << ">";
    Body();
    Out << "</" << Opener.split(' ').first << ">\n";
  }
};

void Writer::writeCode() {
  // This loop (whitespace logic) is cribbed from TokenStream::Print.
  bool FirstToken = true;
  unsigned LastLine = -1;
  StringRef LastText;
  for (const auto &T : Stream.tokens()) {
    StringRef Text = T.text();
    if (FirstToken) {
      FirstToken = false;
    } else if (T.Line == LastLine) {
      if (LastText.data() + LastText.size() != Text.data())
        Out << ' ';
    } else {
      Out << " \n"; // Extra space aids selection.
      Out.indent(T.Indent);
    }
    Out << "<span class='token' id='t" << Stream.index(T) << "'>";
    llvm::printHTMLEscaped(Text, Out);
    Out << "</span>";
    LastLine = T.Line;
    LastText = Text;
  }
  if (!FirstToken)
    Out << '\n';
}

// Writes a JSON array of forest nodes. Items are e.g.:
//   {kind:'sequence', symbol:'compound-stmt', children:[5,8,33],
//   rule:'compound-stmt := ...'} {kind:'terminal', symbol:'VOID', token:'t52'}
//   {kind:'ambiguous', symbol:'type-specifier', children:[3,100] selected:3}
//   {kind:'opaque', symbol:'statement-seq', firstToken:'t5', lastToken:'t6'}
void Writer::writeForestJSON() {
  // This is the flat array of nodes: the index into this array is the node ID.
  std::vector<std::pair<const ForestNode *, /*End*/ Token::Index>> Sequence;
  llvm::DenseMap<const ForestNode *, unsigned> Index;
  auto AssignID = [&](const ForestNode *N, Token::Index End) -> unsigned {
    auto R = Index.try_emplace(N, Sequence.size());
    if (R.second)
      Sequence.push_back({N, End});
    return R.first->second;
  };
  AssignID(&Root, Stream.tokens().size());
  auto TokenID = [](Token::Index I) { return ("t" + llvm::Twine(I)).str(); };

  llvm::json::OStream Out(this->Out, 2);
  Out.array([&] {
    for (unsigned I = 0; I < Sequence.size(); ++I) {
      const ForestNode *N = Sequence[I].first;
      Token::Index End = Sequence[I].second;
      Out.object([&] {
        Out.attribute("symbol", G.symbolName(N->symbol()));
        switch (N->kind()) {
        case ForestNode::Terminal:
          Out.attribute("kind", "terminal");
          Out.attribute("token", TokenID(N->startTokenIndex()));
          break;
        case ForestNode::Sequence:
          Out.attribute("kind", "sequence");
          Out.attribute("rule", G.dumpRule(N->rule()));
          break;
        case ForestNode::Ambiguous:
          Out.attribute("kind", "ambiguous");
          Out.attribute("selected", AssignID(N->children().front(), End));
          break;
        case ForestNode::Opaque:
          Out.attribute("kind", "opaque");
          Out.attribute("firstToken", TokenID(N->startTokenIndex()));
          // [firstToken, lastToken] is a closed range.
          // If empty, lastToken is omitted.
          if (N->startTokenIndex() != End)
            Out.attribute("lastToken", TokenID(End - 1));
          break;
        }
        auto Children = N->children();
        if (!Children.empty())
          Out.attributeArray("children", [&] {
            for (unsigned I = 0; I < Children.size(); ++I)
              Out.value(AssignID(Children[I],
                                 I + 1 == Children.size()
                                     ? End
                                     : Children[I + 1]->startTokenIndex()));
          });
      });
    }
  });
}

} // namespace

// We only accept the derived stream here.
// FIXME: allow the original stream instead?
void writeHTMLForest(llvm::raw_ostream &OS, const Grammar &G,
                     const ForestNode &Root, const TokenStream &Stream) {
  Writer{OS, G, Root, Stream}.write();
}

} // namespace pseudo
} // namespace clang
