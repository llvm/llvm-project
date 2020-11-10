//===- ClangSyntaxEmitter.cpp - Generate clang Syntax Tree nodes ----------===//
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
// In future, the class definitions will be produced by additional backends.
//
//===----------------------------------------------------------------------===//
#include "TableGenBackends.h"

#include <deque>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace {

// The class hierarchy of Node types.
// We assemble this in order to be able to define the NodeKind enum in a
// stable and useful way, where abstract Node subclasses correspond to ranges.
class Hierarchy {
public:
  Hierarchy(const llvm::RecordKeeper &Records) {
    for (llvm::Record *T : Records.getAllDerivedDefinitions("NodeType"))
      add(T);
    for (llvm::Record *Derived : Records.getAllDerivedDefinitions("NodeType"))
      if (llvm::Record *Base = Derived->getValueAsOptionalDef("base"))
        link(Derived, Base);
    for (NodeType &N : AllTypes)
      llvm::sort(N.Derived, [](const NodeType *L, const NodeType *R) {
        return L->Record->getName() < R->Record->getName();
      });
  }

  struct NodeType {
    const llvm::Record *Record = nullptr;
    const NodeType *Base = nullptr;
    std::vector<const NodeType *> Derived;
    llvm::StringRef name() const { return Record->getName(); }
  };

  NodeType &get(llvm::StringRef Name = "Node") {
    auto NI = ByName.find(Name);
    assert(NI != ByName.end() && "no such node");
    return *NI->second;
  }

private:
  void add(const llvm::Record *R) {
    AllTypes.emplace_back();
    AllTypes.back().Record = R;
    assert(ByName.try_emplace(R->getName(), &AllTypes.back()).second &&
           "Duplicate node name");
  }

  void link(const llvm::Record *Derived, const llvm::Record *Base) {
    auto &CN = get(Derived->getName()), &PN = get(Base->getName());
    assert(CN.Base == nullptr && "setting base twice");
    PN.Derived.push_back(&CN);
    CN.Base = &PN;
  }

  std::deque<NodeType> AllTypes;
  llvm::DenseMap<llvm::StringRef, NodeType *> ByName;
};

const Hierarchy::NodeType &firstConcrete(const Hierarchy::NodeType &N) {
  return N.Derived.empty() ? N : firstConcrete(*N.Derived.front());
}
const Hierarchy::NodeType &lastConcrete(const Hierarchy::NodeType &N) {
  return N.Derived.empty() ? N : lastConcrete(*N.Derived.back());
}

void emitNodeList(const Hierarchy::NodeType &N, llvm::raw_ostream &OS) {
  // Don't emit ABSTRACT_NODE for node itself, which has no parent.
  if (N.Base != nullptr) {
    if (N.Derived.empty())
      OS << llvm::formatv("CONCRETE_NODE({0},{1})\n", N.name(), N.Base->name());
    else
      OS << llvm::formatv("ABSTRACT_NODE({0},{1},{2},{3})\n", N.name(),
                          N.Base->name(), firstConcrete(N).name(),
                          lastConcrete(N).name());
  }
  for (const auto *C : N.Derived)
    emitNodeList(*C, OS);
}

} // namespace

void clang::EmitClangSyntaxNodeList(llvm::RecordKeeper &Records,
                                    llvm::raw_ostream &OS) {
  llvm::emitSourceFileHeader("Syntax tree node list", OS);
  OS << "// Generated from " << Records.getInputFilename() << "\n";
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
  emitNodeList(Hierarchy(Records).get(), OS);
  OS << R"cpp(
#undef NODE
#undef CONCRETE_NODE
#undef ABSTRACT_NODE
)cpp";
}
