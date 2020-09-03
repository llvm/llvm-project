//=== ClangASTNodesEmitter.cpp - Generate Clang AST node tables -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang AST node tables
//
//===----------------------------------------------------------------------===//

#include "ASTTableGen.h"
#include "TableGenBackends.h"

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cctype>
#include <map>
#include <set>
#include <string>
using namespace llvm;
using namespace clang;
using namespace clang::tblgen;

/// ClangASTNodesEmitter - The top-level class emits .inc files containing
///  declarations of Clang statements.
///
namespace {
class ClangASTNodesEmitter {
  // A map from a node to each of its derived nodes.
  typedef std::multimap<ASTNode, ASTNode> ChildMap;
  typedef ChildMap::const_iterator ChildIterator;

  RecordKeeper &Records;
  ASTNode Root;
  const std::string &NodeClassName;
  const std::string &BaseSuffix;
  std::string MacroHierarchyName;
  ChildMap Tree;

  // Create a macro-ized version of a name
  static std::string macroName(std::string S) {
    for (unsigned i = 0; i < S.size(); ++i)
      S[i] = std::toupper(S[i]);

    return S;
  }

  const std::string &macroHierarchyName() {
    assert(Root && "root node not yet derived!");
    if (MacroHierarchyName.empty())
      MacroHierarchyName = macroName(std::string(Root.getName()));
    return MacroHierarchyName;
  }

  // Return the name to be printed in the base field. Normally this is
  // the record's name plus the base suffix, but if it is the root node and
  // the suffix is non-empty, it's just the suffix.
  std::string baseName(ASTNode node) {
    if (node == Root && !BaseSuffix.empty())
      return BaseSuffix;

    return node.getName().str() + BaseSuffix;
  }

  void deriveChildTree();

  std::pair<ASTNode, ASTNode> EmitNode(raw_ostream& OS, ASTNode Base);
public:
  explicit ClangASTNodesEmitter(RecordKeeper &R, const std::string &N,
                                const std::string &S)
    : Records(R), NodeClassName(N), BaseSuffix(S) {}

  // run - Output the .inc file contents
  void run(raw_ostream &OS);
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Statement Node Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

// Returns the first and last non-abstract subrecords
// Called recursively to ensure that nodes remain contiguous
std::pair<ASTNode, ASTNode> ClangASTNodesEmitter::EmitNode(raw_ostream &OS,
                                                           ASTNode Base) {
  std::string BaseName = macroName(std::string(Base.getName()));

  ChildIterator i = Tree.lower_bound(Base), e = Tree.upper_bound(Base);
  bool HasChildren = (i != e);

  ASTNode First, Last;
  if (!Base.isAbstract())
    First = Last = Base;

  for (; i != e; ++i) {
    ASTNode Child = i->second;
    bool Abstract = Child.isAbstract();
    std::string NodeName = macroName(std::string(Child.getName()));

    OS << "#ifndef " << NodeName << "\n";
    OS << "#  define " << NodeName << "(Type, Base) "
        << BaseName << "(Type, Base)\n";
    OS << "#endif\n";

    if (Abstract) OS << "ABSTRACT_" << macroHierarchyName() << "(";
    OS << NodeName << "(" << Child.getName() << ", " << baseName(Base) << ")";
    if (Abstract) OS << ")";
    OS << "\n";

    auto Result = EmitNode(OS, Child);
    assert(Result.first && Result.second && "node didn't have children?");

    // Update the range of Base.
    if (!First) First = Result.first;
    Last = Result.second;

    OS << "#undef " << NodeName << "\n\n";
  }

  // If there aren't first/last nodes, it must be because there were no
  // children and this node was abstract, which is not a sensible combination.
  if (!First) {
    PrintFatalError(Base.getLoc(), "abstract node has no children");
  }
  assert(Last && "set First without Last");

  if (HasChildren) {
    // Use FOO_RANGE unless this is the last of the ranges, in which case
    // use LAST_FOO_RANGE.
    if (Base == Root)
      OS << "LAST_" << macroHierarchyName() << "_RANGE(";
    else
      OS << macroHierarchyName() << "_RANGE(";
    OS << Base.getName() << ", " << First.getName() << ", "
       << Last.getName() << ")\n\n";
  }

  return std::make_pair(First, Last);
}

void ClangASTNodesEmitter::deriveChildTree() {
  assert(!Root && "already computed tree");

  // Emit statements
  const std::vector<Record*> Stmts
    = Records.getAllDerivedDefinitions(NodeClassName);

  for (unsigned i = 0, e = Stmts.size(); i != e; ++i) {
    Record *R = Stmts[i];

    if (auto B = R->getValueAsOptionalDef(BaseFieldName))
      Tree.insert(std::make_pair(B, R));
    else if (Root)
      PrintFatalError(R->getLoc(),
                      Twine("multiple root nodes in \"") + NodeClassName
                        + "\" hierarchy");
    else
      Root = R;
  }

  if (!Root)
    PrintFatalError(Twine("didn't find root node in \"") + NodeClassName
                      + "\" hierarchy");
}

void ClangASTNodesEmitter::run(raw_ostream &OS) {
  deriveChildTree();

  emitSourceFileHeader("List of AST nodes of a particular kind", OS);

  // Write the preamble
  OS << "#ifndef ABSTRACT_" << macroHierarchyName() << "\n";
  OS << "#  define ABSTRACT_" << macroHierarchyName() << "(Type) Type\n";
  OS << "#endif\n";

  OS << "#ifndef " << macroHierarchyName() << "_RANGE\n";
  OS << "#  define "
     << macroHierarchyName() << "_RANGE(Base, First, Last)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_" << macroHierarchyName() << "_RANGE\n";
  OS << "#  define LAST_" 
     << macroHierarchyName() << "_RANGE(Base, First, Last) " 
     << macroHierarchyName() << "_RANGE(Base, First, Last)\n";
  OS << "#endif\n\n";

  EmitNode(OS, Root);

  OS << "#undef " << macroHierarchyName() << "\n";
  OS << "#undef " << macroHierarchyName() << "_RANGE\n";
  OS << "#undef LAST_" << macroHierarchyName() << "_RANGE\n";
  OS << "#undef ABSTRACT_" << macroHierarchyName() << "\n";
}

void clang::EmitClangASTNodes(RecordKeeper &RK, raw_ostream &OS,
                              const std::string &N, const std::string &S) {
  ClangASTNodesEmitter(RK, N, S).run(OS);
}

// Emits and addendum to a .inc file to enumerate the clang declaration
// contexts.
void clang::EmitClangDeclContext(RecordKeeper &Records, raw_ostream &OS) {
  // FIXME: Find a .td file format to allow for this to be represented better.

  emitSourceFileHeader("List of AST Decl nodes", OS);

  OS << "#ifndef DECL_CONTEXT\n";
  OS << "#  define DECL_CONTEXT(DECL)\n";
  OS << "#endif\n";
  
  OS << "#ifndef DECL_CONTEXT_BASE\n";
  OS << "#  define DECL_CONTEXT_BASE(DECL) DECL_CONTEXT(DECL)\n";
  OS << "#endif\n";
  
  typedef std::set<Record*> RecordSet;
  typedef std::vector<Record*> RecordVector;
  
  RecordVector DeclContextsVector
    = Records.getAllDerivedDefinitions(DeclContextNodeClassName);
  RecordVector Decls = Records.getAllDerivedDefinitions(DeclNodeClassName);
  RecordSet DeclContexts (DeclContextsVector.begin(), DeclContextsVector.end());
   
  for (RecordVector::iterator i = Decls.begin(), e = Decls.end(); i != e; ++i) {
    Record *R = *i;

    if (Record *B = R->getValueAsOptionalDef(BaseFieldName)) {
      if (DeclContexts.find(B) != DeclContexts.end()) {
        OS << "DECL_CONTEXT_BASE(" << B->getName() << ")\n";
        DeclContexts.erase(B);
      }
    }
  }

  // To keep identical order, RecordVector may be used
  // instead of RecordSet.
  for (RecordVector::iterator
         i = DeclContextsVector.begin(), e = DeclContextsVector.end();
       i != e; ++i)
    if (DeclContexts.find(*i) != DeclContexts.end())
      OS << "DECL_CONTEXT(" << (*i)->getName() << ")\n";

  OS << "#undef DECL_CONTEXT\n";
  OS << "#undef DECL_CONTEXT_BASE\n";
}
