//===- DependencyScanner.h - Module dependency discovery --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the module dependency graph and dependency-scanning
/// functionality.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H
#define LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Allocator.h"

namespace llvm::opt {
class DerivedArgList;
} // namespace llvm::opt

namespace clang {
class DiagnosticsEngine;
namespace driver {
class Driver;
} // namespace driver
} // namespace clang

namespace clang::driver::dependencies {

using ClangModuleDeps = tooling::dependencies::ModuleDeps;
using ClangModuleGraph = tooling::dependencies::ModuleDepsGraph;
using tooling::dependencies::ModuleID;
using tooling::dependencies::TranslationUnitDeps;

//===----------------------------------------------------------------------===//
// Dependency Scan
//===----------------------------------------------------------------------===//

/// Represents the dependency scanning result for a single source input.
struct TranslationUnitScanResult {
  TranslationUnitScanResult(size_t JobIndex, std::string &&Filename,
                            TranslationUnitDeps &&TUDeps)
      : JobIndex(JobIndex), Filename(std::move(Filename)),
        TUDeps(std::move(TUDeps)) {}
  /// Index of the job associated with this scan result, which the driver will
  /// later create.
  size_t JobIndex;

  /// The source input for which the scan was run.
  std::string Filename;

  /// The full dependencies and Clang module graph for this input.
  TranslationUnitDeps TUDeps;
};

/// Computes module dependencies for the given driver command line.
///
/// \param ClangExecutable - The path to the main clang executable.
/// \param Diags - The driver's diagnostics engine.
/// \param Args - The driver's command line.
///
/// \returns A vector of scan results (one per scannable source input), or an
/// error if any input fails to scan. The order of scan results is
/// deterministic.
llvm::Expected<llvm::SmallVector<TranslationUnitScanResult, 0>>
scanDependencies(llvm::StringRef ClangExecutable,
                 clang::DiagnosticsEngine &Diags,
                 const llvm::opt::DerivedArgList &Args);

//===----------------------------------------------------------------------===//
// Module Dependency Graph
//===----------------------------------------------------------------------===//

class MDGNode;
class MDGEdge;
using MDGNodeBase = llvm::DGNode<MDGNode, MDGEdge>;
using MDGEdgeBase = llvm::DGEdge<MDGNode, MDGEdge>;
using ModuleDepGraphBase = llvm::DirectedGraph<MDGNode, MDGEdge>;

/// Abstract base class for all node kinds in the module dependency graph.
class MDGNode : public MDGNodeBase {
public:
  enum class NodeKind {
    Root,
    ClangModule,
    NamedCXXModule,
    NonModule,
  };

  explicit MDGNode(NodeKind Kind) : Kind(Kind) {}

  /// Returns this node's kind.
  NodeKind getKind() const { return Kind; }

  /// Returns the list of Clang modules this module/translation unit directly
  /// depends on.
  virtual llvm::ArrayRef<ModuleID> getClangModuleDeps() const { return {}; }

  /// Returns the list of C++20 named modules this translation unit directly
  /// depends on.
  virtual llvm::ArrayRef<std::string> getCXXNamedModuleDeps() const {
    return {};
  }

protected:
  virtual ~MDGNode() = 0;

private:
  const NodeKind Kind;
};

/// Represents the root node of the module dependency graph.
///
/// The root node only serves as an entry point for graph traversal.
/// It should have an edge to each node that would otherwise have no incoming
/// edges, ensuring there is always a path from the root to any node in the
/// graph.
/// There should be exactly one such root node in a given graph.
class RootMDGNode final : public MDGNode {
public:
  RootMDGNode() : MDGNode(NodeKind::Root) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::Root;
  }
};

/// Base class defining common functionality for nodes that represent a
/// translation unit from a command line source input.
class SourceInputBackedMDGNode : public MDGNode {
protected:
  explicit SourceInputBackedMDGNode(
      const NodeKind Kind, const TranslationUnitScanResult &BackingScanResult)
      : MDGNode(Kind), BackingScanResult(&BackingScanResult) {}

  /// The backing scan result, owned by the module dependency graph.
  const TranslationUnitScanResult *BackingScanResult;

public:
  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    auto K = N->getKind();
    return K == NodeKind::NonModule || K == NodeKind::NamedCXXModule;
  }

  /// Returns the path of this translation unit's main source file.
  llvm::StringRef getFilename() const { return BackingScanResult->Filename; }

  /// Returns the index of the -cc1 driver job for this translation unit, which
  /// the driver will later create.
  size_t getJobIndex() const { return BackingScanResult->JobIndex; }

  llvm::ArrayRef<std::string> getCXXNamedModuleDeps() const override {
    return BackingScanResult->TUDeps.NamedModuleDeps;
  }

  llvm::ArrayRef<ModuleID> getClangModuleDeps() const override {
    return BackingScanResult->TUDeps.ClangModuleDeps;
  }
};

/// Subclass of MDGNode representing a translation unit that does not provide
/// any module.
class NonModuleMDGNode final : public SourceInputBackedMDGNode {
public:
  explicit NonModuleMDGNode(const TranslationUnitScanResult &BackingScanResult)
      : SourceInputBackedMDGNode(NodeKind::NonModule, BackingScanResult) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::NonModule;
  }
};

/// Subclass of MDGNode representing a translation unit that provides a C++20
/// named module.
class NamedCXXModuleMDGNode final : public SourceInputBackedMDGNode {
public:
  explicit NamedCXXModuleMDGNode(
      const TranslationUnitScanResult &BackingScanResult)
      : SourceInputBackedMDGNode(NodeKind::NamedCXXModule, BackingScanResult) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::NamedCXXModule;
  }

  /// Returns the unique identifier for the named C++ module this translation
  /// unit exports.
  const ModuleID &getModuleID() const { return BackingScanResult->TUDeps.ID; }
};

/// Subclass of MDGNode representing a Clang module unit.
class ClangModuleMDGNode final : public MDGNode {
public:
  explicit ClangModuleMDGNode(const ClangModuleDeps &BackingModuleDeps)
      : MDGNode(NodeKind::ClangModule), BackingModuleDeps(&BackingModuleDeps) {}

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::ClangModule;
  }

  /// Returns the unique identifier for this Clang module.
  const ModuleID &getModuleID() const { return BackingModuleDeps->ID; }

  /// Returns the list of Clang modules this module unit directly depends on.
  llvm::ArrayRef<ModuleID> getClangModuleDeps() const override {
    return BackingModuleDeps->ClangModuleDeps;
  }

private:
  /// The backing scan result, owned by the module dependency graph.
  const ClangModuleDeps *BackingModuleDeps;
};

/// Represents an import relation in the module dependency graph, directed from
/// the imported module to the importer.
class MDGEdge : public MDGEdgeBase {
public:
  explicit MDGEdge(MDGNode &N) : MDGEdgeBase(N) {}
  MDGEdge() = delete;
};

namespace detail {
class ModuleDepGraphBuilder;
} // namespace detail

/// A directed graph describing module import relationships.
///
/// The graph owns its nodes, edges, and the dependency scan results from which
/// it was created.
/// Non-root nodes provide a view into the backing scan results.
class ModuleDepGraph : public ModuleDepGraphBase {
public:
  explicit ModuleDepGraph(
      llvm::SmallVectorImpl<TranslationUnitScanResult> &&ScanResults)
      : BackingScanResults(std::move(ScanResults)) {}

  MDGNode *getRoot() { return Root; }

  const MDGNode *getRoot() const { return Root; }

private:
  friend class detail::ModuleDepGraphBuilder;

  llvm::BumpPtrAllocator BumpPtrAlloc;
  llvm::SmallVector<TranslationUnitScanResult, 0> BackingScanResults;
  RootMDGNode *Root = nullptr;
};

/// Build a module dependency graph from the given \c ScanResults.
///
/// \returns The constructed graph, or an error if conflicting module
/// definitions are found.
llvm::Expected<ModuleDepGraph>
buildModuleDepGraph(llvm::SmallVectorImpl<TranslationUnitScanResult> &&Scans,
                    DiagnosticsEngine &Diags);

/// Writes the module dependency graph to the given output stream.
void writeModuleDepGraph(raw_ostream &OS, const ModuleDepGraph &G);

} // namespace clang::driver::dependencies

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphTraits specializations
//===----------------------------------------------------------------------===//

namespace llvm {
/// Non-const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<clang::driver::dependencies::MDGNode *> {
  using NodeTy = clang::driver::dependencies::MDGNode;
  using NodeRef = NodeTy *;

  static NodeRef MDGGetTargetNode(clang::driver::dependencies::MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  // Provide a mapped iterator so that GraphTraits-based implementations can
  // find the target nodes without explicitly going through the edges.
  using ChildIteratorType =
      mapped_iterator<NodeTy::iterator, decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = NodeTy::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &MDGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &MDGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<clang::driver::dependencies::ModuleDepGraph *>
    : GraphTraits<clang::driver::dependencies::MDGNode *> {
  using GraphTy = clang::driver::dependencies::ModuleDepGraph;
  using GraphRef = GraphTy *;
  using NodeRef = clang::driver::dependencies::MDGNode *;

  using nodes_iterator = GraphTy::iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

/// Const versions of the GraphTraits specializations for ModuleDepGraph.
template <> struct GraphTraits<const clang::driver::dependencies::MDGNode *> {
  using NodeTy = const clang::driver::dependencies::MDGNode;
  using NodeRef = NodeTy *;

  static NodeRef
  MDGGetTargetNode(const clang::driver::dependencies::MDGEdgeBase *E) {
    return &E->getTargetNode();
  }

  // Provide a mapped iterator so that GraphTraits-based implementations can
  // find the target nodes without explicitly going through the edges.
  using ChildIteratorType =
      mapped_iterator<NodeTy::const_iterator, decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = NodeTy::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &MDGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &MDGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<const clang::driver::dependencies::ModuleDepGraph *>
    : GraphTraits<const clang::driver::dependencies::MDGNode *> {
  using GraphTy = const clang::driver::dependencies::ModuleDepGraph;
  using GraphRef = GraphTy *;
  using NodeRef = const clang::driver::dependencies::MDGNode *;

  using nodes_iterator = GraphTy::const_iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};
} // namespace llvm

#endif
