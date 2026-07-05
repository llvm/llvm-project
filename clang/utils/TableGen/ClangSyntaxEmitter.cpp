//===-- ClangSyntaxEmitter.cpp - Generate clang Syntax Tree nodes ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These backends consume the definitions of Syntax Tree nodes.
// See clang/include/clang/Tooling/Syntax/{Syntax,Nodes}.td
//
// The -gen-clang-syntax-node-list backend produces a .inc with macro calls
//   NODE(Kind, BaseKind)
//   ABSTRACT_NODE(Type, Base, FirstKind, LastKind)
// similar to those for AST nodes such as AST/DeclNodes.inc.
//
// The -gen-clang-syntax-node-classes backend produces definitions for the
// syntax::Node subclasses (except those marked as External).
//
// In future, another backend will encode the structure of the various node
// types in tables so their invariants can be checked and enforced.
//
//===----------------------------------------------------------------------===//
#include "TableGenBackends.h"

#include <deque>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

// The class hierarchy of Node types.
// We assemble this in order to be able to define the NodeKind enum in a
// stable and useful way, where abstract Node subclasses correspond to ranges.
class Hierarchy {
public:
  Hierarchy(const RecordKeeper &Records) {
    for (const Record *T : Records.getAllDerivedDefinitions("NodeType"))
      add(T);
    for (const Record *Derived : Records.getAllDerivedDefinitions("NodeType"))
      if (const Record *Base = Derived->getValueAsOptionalDef("base"))
        link(Derived, Base);
    for (NodeType &N : AllTypes) {
      sort(N.Derived, [](const NodeType *L, const NodeType *R) {
        return L->Rec->getName() < R->Rec->getName();
      });
      // Alternatives nodes must have subclasses, External nodes may do.
      assert(N.Rec->isSubClassOf("Alternatives") ||
             N.Rec->isSubClassOf("External") || N.Derived.empty());
      assert(!N.Rec->isSubClassOf("Alternatives") || !N.Derived.empty());
    }
  }

  struct NodeType {
    const Record *Rec = nullptr;
    const NodeType *Base = nullptr;
    std::vector<const NodeType *> Derived;
    StringRef name() const { return Rec->getName(); }
  };

  NodeType &get(StringRef Name = "Node") {
    auto NI = ByName.find(Name);
    assert(NI != ByName.end() && "no such node");
    return *NI->second;
  }

  // Traverse the hierarchy in pre-order (base classes before derived).
  void visit(function_ref<void(const NodeType &)> CB,
             const NodeType *Start = nullptr) {
    if (Start == nullptr)
      Start = &get();
    CB(*Start);
    for (const NodeType *D : Start->Derived)
      visit(CB, D);
  }

private:
  void add(const Record *R) {
    AllTypes.emplace_back();
    AllTypes.back().Rec = R;
    bool Inserted = ByName.try_emplace(R->getName(), &AllTypes.back()).second;
    assert(Inserted && "Duplicate node name");
    (void)Inserted;
  }

  void link(const Record *Derived, const Record *Base) {
    auto &CN = get(Derived->getName()), &PN = get(Base->getName());
    assert(CN.Base == nullptr && "setting base twice");
    PN.Derived.push_back(&CN);
    CN.Base = &PN;
  }

  std::deque<NodeType> AllTypes;
  DenseMap<StringRef, NodeType *> ByName;
};

const Hierarchy::NodeType &firstConcrete(const Hierarchy::NodeType &N) {
  return N.Derived.empty() ? N : firstConcrete(*N.Derived.front());
}
const Hierarchy::NodeType &lastConcrete(const Hierarchy::NodeType &N) {
  return N.Derived.empty() ? N : lastConcrete(*N.Derived.back());
}

struct SyntaxConstraint {
  SyntaxConstraint(const Record &R) {
    if (R.isSubClassOf("Optional")) {
      *this = SyntaxConstraint(*R.getValueAsDef("inner"));
    } else if (R.isSubClassOf("AnyToken")) {
      NodeType = "Leaf";
    } else if (R.isSubClassOf("NodeType")) {
      NodeType = R.getName();
    } else {
      assert(false && "Unhandled Syntax kind");
    }
  }

  StringRef NodeType;
  // optional and leaf types also go here, once we want to use them.
};

} // namespace

void clang::EmitClangSyntaxNodeList(const RecordKeeper &Records,
                                    raw_ostream &OS) {
  emitSourceFileHeader("Syntax tree node list", OS, Records);
  Hierarchy H(Records);
  OS << R"cpp(
#ifndef NODE
#define NODE(Kind, Base)
#endif

#ifndef CONCRETE_NODE
#define CONCRETE_NODE(Kind, Base) NODE(Kind, Base)
#endif

#ifndef ABSTRACT_NODE
#define ABSTRACT_NODE(Kind, Base, First, Last) NODE(Kind, Base)
#endif

)cpp";
  H.visit([&](const Hierarchy::NodeType &N) {
    // Don't emit ABSTRACT_NODE for node itself, which has no parent.
    if (N.Base == nullptr)
      return;
    if (N.Derived.empty())
      OS << formatv("CONCRETE_NODE({0},{1})\n", N.name(), N.Base->name());
    else
      OS << formatv("ABSTRACT_NODE({0},{1},{2},{3})\n", N.name(),
                    N.Base->name(), firstConcrete(N).name(),
                    lastConcrete(N).name());
  });
  OS << R"cpp(
#undef NODE
#undef CONCRETE_NODE
#undef ABSTRACT_NODE
)cpp";
}

// Format a documentation string as a C++ comment.
// Trims leading whitespace handling since comments come from a TableGen file:
//    documentation = [{
//      This is a widget. Example:
//        widget.explode()
//    }];
// and should be formatted as:
//    /// This is a widget. Example:
//    ///   widget.explode()
// Leading and trailing whitespace lines are stripped.
// The indentation of the first line is stripped from all lines.
static void printDoc(StringRef Doc, raw_ostream &OS) {
  Doc = Doc.rtrim();
  StringRef Line;
  while (Line.trim().empty() && !Doc.empty())
    std::tie(Line, Doc) = Doc.split('\n');
  StringRef Indent = Line.take_while(isSpace);
  for (; !Line.empty() || !Doc.empty(); std::tie(Line, Doc) = Doc.split('\n')) {
    Line.consume_front(Indent);
    OS << "/// " << Line << "\n";
  }
}

void clang::EmitClangSyntaxNodeClasses(const RecordKeeper &Records,
                                       raw_ostream &OS) {
  emitSourceFileHeader("Syntax tree node list", OS, Records);
  Hierarchy H(Records);

  OS << "\n// Forward-declare node types so we don't have to carefully "
        "sequence definitions.\n";
  H.visit([&](const Hierarchy::NodeType &N) {
    OS << "class " << N.name() << ";\n";
  });

  OS << "\n// Node definitions\n\n";
  H.visit([&](const Hierarchy::NodeType &N) {
    if (N.Rec->isSubClassOf("External"))
      return;
    printDoc(N.Rec->getValueAsString("documentation"), OS);
    OS << formatv("class {0}{1} : public {2} {{\n", N.name(),
                  N.Derived.empty() ? " final" : "", N.Base->name());

    // Constructor.
    if (N.Derived.empty())
      OS << formatv("public:\n  {0}() : {1}(NodeKind::{0}) {{}\n", N.name(),
                    N.Base->name());
    else
      OS << formatv("protected:\n  {0}(NodeKind K) : {1}(K) {{}\npublic:\n",
                    N.name(), N.Base->name());

    if (N.Rec->isSubClassOf("Sequence")) {
      // Getters for sequence elements.
      for (const auto &C : N.Rec->getValueAsListOfDefs("children")) {
        assert(C->isSubClassOf("Role"));
        StringRef Role = C->getValueAsString("role");
        SyntaxConstraint Constraint(*C->getValueAsDef("syntax"));
        for (const char *Const : {"", "const "})
          OS << formatv(
              "  {2}{1} *get{0}() {2} {{\n"
              "    return llvm::cast_or_null<{1}>(findChild(NodeRole::{0}));\n"
              "  }\n",
              Role, Constraint.NodeType, Const);
      }
    }

    // classof. FIXME: move definition inline once ~all nodes are generated.
    OS << "  static bool classof(const Node *N);\n";

    OS << "};\n\n";
  });
}
