#ifndef LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H
#define LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H

#include "clang/Driver/Driver.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/GraphWriter.h"

namespace clang {
class SourceManager;
class CharSourceRange;
class DiagnosticsEngine;
} // namespace clang

namespace llvm::opt {
class DerivedArgList;
} // namespace llvm::opt

namespace clang {
namespace driver {
namespace dependencies {

using clang::tooling::dependencies::TranslationUnitDeps;

//===----------------------------------------------------------------------===//
// Dependency Scan
//===----------------------------------------------------------------------===//

class DependencyScanError : public llvm::ErrorInfo<DependencyScanError> {
public:
  static char ID;

  void log(llvm::raw_ostream &OS) const override {
    OS << "error while performing dependency scan\n";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::errc::not_supported;
  }
};

/// Performs a full dependency scan for the given driver command line and
/// returns all scan results or an error on scanning failure.
///
/// \param ClangProgramPath Path to the clang executable
/// \param Diags            The calling driver's diagnostics engine
/// \param Args             The calling driver's command line arguments
///
/// \returns The scan results for all inputs, or an error if scanning fails
llvm::Expected<SmallVector<TranslationUnitDeps, 0>>
scanModuleDependencies(llvm::StringRef ClangProgramPath,
                       clang::DiagnosticsEngine &Diags,
                       const llvm::opt::DerivedArgList &Args);

//===----------------------------------------------------------------------===//
// Module Dependency Graph
//===----------------------------------------------------------------------===//

class MDGNode;
class MDGEdge;
using MDGNodeBase = llvm::DGNode<MDGNode, MDGEdge>;
using MDGEdgeBase = llvm::DGEdge<MDGNode, MDGEdge>;
using MDGBase = llvm::DirectedGraph<MDGNode, MDGEdge>;

/// Base class for module dependency graph nodes.
///
/// Represents a node in the ModuleDepGraph, which can be a translation unit
/// which doesn't provide any module, a Clang module, or a C++ named module.
class MDGNode : public MDGNodeBase {
public:
  enum class NodeKind {
    ClangModule,
    NonModuleTU,
    CXXNamedModule,
  };

  using Command = tooling::dependencies::Command;

  MDGNode(const NodeKind K) : Kind(K) {}
  MDGNode(const NodeKind K, std::vector<Command> Commands)
      : Kind(K), Commands(Commands) {}

  virtual ~MDGNode() = 0;

  NodeKind getKind() const { return Kind; }

  ArrayRef<Command> getCommands() const { return Commands; }

private:
  const NodeKind Kind;

protected:
  std::vector<Command> Commands;
};

/// ClangModuleNode - represents a Clang module in the ModuleDepGraph.
class ClangModuleNode : public MDGNode {
public:
  ClangModuleNode(StringRef ModuleName)
      : MDGNode(NodeKind::ClangModule), ModuleName(ModuleName) {}
  ~ClangModuleNode() = default;

  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::ClangModule;
  }

  StringRef getModuleName() const { return ModuleName; }

  void setCommands(std::vector<tooling::dependencies::Command> Commands) {
    this->Commands = Commands;
  }

private:
  std::string ModuleName;
};

/// NonModuleTUNode - represents a regular TU which doesn't provide any module,
/// in the ModuleDepGraph.
class NonModuleTUNode : public MDGNode {
public:
  NonModuleTUNode(StringRef InputFile)
      : MDGNode(NodeKind::NonModuleTU), InputFile(InputFile) {}
  ~NonModuleTUNode() override = default;

  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::NonModuleTU;
  }

  StringRef getInputFile() const { return InputFile; }

private:
  const std::string InputFile;
};

/// CXXNamedModuleNode - represents a C++ named module node in the
/// ModuleDepGraph.
///
/// Unresolved nodes are those discovered as imports but missing a module
/// definition.
class CXXNamedModuleNode : public MDGNode {
public:
  CXXNamedModuleNode(StringRef ModuleName, StringRef InputFile = "")
      : MDGNode(NodeKind::CXXNamedModule), InputFile(InputFile),
        ModuleName(ModuleName) {}
  ~CXXNamedModuleNode() = default;

  static bool classof(const MDGNode *N) {
    return N->getKind() == NodeKind::CXXNamedModule;
  }

  StringRef getInputFile() const { return InputFile; }

  StringRef getModuleName() const { return ModuleName; }

  bool isUnresolved() const { return InputFile.empty(); }

  void setInputFile(StringRef FilePath) { this->InputFile = FilePath; }

  void setCommands(std::vector<tooling::dependencies::Command> Commands) {
    this->Commands = Commands;
  }

private:
  std::string InputFile;
  std::string ModuleName;
};

/// MDGEdge - represents an import relationship, directed from the importing
/// unit to the imported unit.
class MDGEdge : public MDGEdgeBase {
public:
  MDGEdge() = delete;
  MDGEdge(MDGNode &N) : MDGEdgeBase(N) {}
};

class ModuleDepGraphBuilder;

/// ModuleDepGraph - A directed graph that represents dependency relationships
/// from dependency scan results, with ownership of its nodes and edges.
class ModuleDepGraph : public MDGBase {
  friend ModuleDepGraphBuilder;

  template <typename NodeTy, typename... Args>
  NodeTy *MakeWithBumpAlloc(Args &&...args);

  llvm::BumpPtrAllocator BumpPtrAlloc;
};

/// Fully constructs a ModuleDepGraph from the dependency scan results.
///
/// \param ScanResults The list of scan results.
/// \param Inputs The calling drivers list of input list in its original order.
/// \param Path to the clang executable.
ModuleDepGraph
buildModuleDepGraph(SmallVectorImpl<TranslationUnitDeps> &&ScanResults,
                    clang::driver::Driver::InputList Inputs,
                    StringRef ClangProgramPath);

} // namespace dependencies
} // namespace driver
} // namespace clang

//===----------------------------------------------------------------------===//
// Module Dependency Graph: GraphTraits specialization
//===----------------------------------------------------------------------===//

namespace llvm {
/// non-const versions of the GraphTrait specializations for MDG
template <> struct GraphTraits<clang::driver::dependencies::MDGNode *> {
  using NodeRef = clang::driver::dependencies::MDGNode *;

  static NodeRef MDGGetTargetNode(clang::driver::dependencies::MDGEdgeBase *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations can
  // find the target nodes without having to explicitly go through the edges.
  using ChildIteratorType =
      mapped_iterator<clang::driver::dependencies::MDGNode::iterator,
                      decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType = clang::driver::dependencies::MDGNode::iterator;

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
    : public GraphTraits<clang::driver::dependencies::MDGNode *> {
  using nodes_iterator = clang::driver::dependencies::ModuleDepGraph::iterator;
  static NodeRef getEntryNode(clang::driver::dependencies::ModuleDepGraph *G) {
    return *G->begin();
  }
  static nodes_iterator
  nodes_begin(clang::driver::dependencies::ModuleDepGraph *G) {
    return G->begin();
  }
  static nodes_iterator
  nodes_end(clang::driver::dependencies::ModuleDepGraph *G) {
    return G->end();
  }
};

/// const versions of the GraphTrait specializations for MDG
template <> struct GraphTraits<const clang::driver::dependencies::MDGNode *> {
  using NodeRef = const clang::driver::dependencies::MDGNode *;

  static NodeRef
  MDGGetTargetNode(const clang::driver::dependencies::MDGEdgeBase *P) {
    return &P->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<clang::driver::dependencies::MDGNode::const_iterator,
                      decltype(&MDGGetTargetNode)>;
  using ChildEdgeIteratorType =
      clang::driver::dependencies::MDGNode::const_iterator;

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
    : public GraphTraits<const clang::driver::dependencies::MDGNode *> {
  using nodes_iterator =
      clang::driver::dependencies::ModuleDepGraph::const_iterator;

  static NodeRef
  getEntryNode(const clang::driver::dependencies::ModuleDepGraph *G) {
    return *G->begin();
  }
  static nodes_iterator
  nodes_begin(const clang::driver::dependencies::ModuleDepGraph *G) {
    return G->begin();
  }
  static nodes_iterator
  nodes_end(const clang::driver::dependencies::ModuleDepGraph *G) {
    return G->end();
  }
};

//===----------------------------------------------------------------------===//
// Module Dependency Graph: DOTGraphTraits & GraphWriter specializations
//===----------------------------------------------------------------------===//

template <>
struct DOTGraphTraits<const clang::driver::dependencies::ModuleDepGraph *>
    : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static StringRef
  getGraphName(const clang::driver::dependencies::ModuleDepGraph *MDG) {
    return "Module Dependency Graph";
  }

  static StringRef
  getNodeKindLabel(const clang::driver::dependencies::NonModuleTUNode *N) {
    return "Non-Module TU";
  }

  static StringRef
  getNodeKindLabel(const clang::driver::dependencies::ClangModuleNode *N) {
    return "Clang Module";
  }

  static StringRef
  getNodeKindLabel(const clang::driver::dependencies::CXXNamedModuleNode *N) {
    return "C++ Named Module";
  }

  static std::string
  getNodeIdentifierLabel(const clang::driver::dependencies::MDGNode *N,
                         const clang::driver::dependencies::ModuleDepGraph *G) {
    using namespace clang::driver::dependencies;
    if (const auto *ClangModule = dyn_cast<ClangModuleNode>(N))
      return (Twine(getNodeKindLabel(ClangModule)) + " '" +
              ClangModule->getModuleName() + "'")
          .str();
    if (const auto *CXXNamedModule = dyn_cast<CXXNamedModuleNode>(N))
      return (Twine(getNodeKindLabel(CXXNamedModule)) + " '" +
              CXXNamedModule->getModuleName() + "'")
          .str();
    if (const auto *NonModuleTU = dyn_cast<NonModuleTUNode>(N))
      return (Twine(getNodeKindLabel(NonModuleTU)) + " '" +
              NonModuleTU->getInputFile() + "'")
          .str();
    llvm_unreachable("Unhandled MDGNode kind!");
  }

  static std::string
  getGraphProperties(const clang::driver::dependencies::ModuleDepGraph *G) {
    return "\tnode [shape=Mrecord];\n\tedge [dir=\"back\"];\n";
  }
};

template <>
class GraphWriter<const clang::driver::dependencies::ModuleDepGraph *>
    : public GraphWriterBase<
          const clang::driver::dependencies::ModuleDepGraph *,
          GraphWriter<const clang::driver::dependencies::ModuleDepGraph *>> {
public:
  using GraphType = const clang::driver::dependencies::ModuleDepGraph *;
  using Base = GraphWriterBase<GraphType, GraphWriter<GraphType>>;

  GraphWriter(raw_ostream &o, const GraphType &g, bool SN) : Base(o, g, SN) {}

  void writeNodes();

private:
  using Base::DOTTraits;
  using Base::GTraits;
  using Base::NodeRef;

  void writeNodeDeclarations(ArrayRef<NodeRef> Nodes);
  void writeNodeDeclaration(NodeRef Node);
  void writeNodeRelations(ArrayRef<NodeRef> Nodes);
  void writeNodeRelation(NodeRef Node);

  DenseMap<NodeRef, std::string> NodeIDLabels;
};

} // namespace llvm

#endif
