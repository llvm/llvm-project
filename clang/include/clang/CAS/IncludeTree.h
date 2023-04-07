//===- IncludeTree.h - Include-tree CAS graph -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CAS_CASINCLUDETREE_H
#define LLVM_CLANG_CAS_CASINCLUDETREE_H

#include "clang/Basic/SourceManager.h"
#include "llvm/CAS/ObjectStore.h"

namespace llvm {
class SmallBitVector;
}

namespace clang {
namespace cas {

/// Base class for include-tree related nodes. It makes it convenient to
/// add/skip/check the "node kind identifier" (\p getNodeKind()) that is put
/// at the beginning of the object data for every include-tree related node.
template <typename NodeT> class IncludeTreeBase : public ObjectProxy {
protected:
  explicit IncludeTreeBase(ObjectProxy Node) : ObjectProxy(std::move(Node)) {
    assert(isValid(*this));
  }

  StringRef getData() const {
    return ObjectProxy::getData().substr(NodeT::getNodeKind().size());
  }

  static Expected<NodeT> create(ObjectStore &DB, ArrayRef<ObjectRef> Refs,
                                ArrayRef<char> Data);

  static bool isValid(const ObjectProxy &Node) {
    return Node.getData().startswith(NodeT::getNodeKind());
  }

  friend NodeT;
};

/// Represents a DAG of included files by the preprocessor and module imports.
/// Each node in the DAG represents a particular inclusion of a file or module
/// that encompasses inclusions of other files as sub-trees, along with all the
/// \p __has_include() preprocessor checks that occurred during preprocessing
/// of that file.
class IncludeTree : public IncludeTreeBase<IncludeTree> {
public:
  static constexpr StringRef getNodeKind() { return "Tree"; }

  class File;
  class FileList;
  class Node;
  class Module;
  class ModuleImport;
  class ModuleMap;

  Expected<File> getBaseFile();

  /// The include file that resulted in this include-tree.
  ObjectRef getBaseFileRef() const { return getReference(0); }

  struct FileInfo {
    StringRef Filename;
    StringRef Contents;
  };

  Expected<FileInfo> getBaseFileInfo();

  SrcMgr::CharacteristicKind getFileCharacteristic() const {
    return (SrcMgr::CharacteristicKind)(getData().front() & ~IsSubmoduleBit);
  }

  bool isSubmodule() const { return (getData().front() & IsSubmoduleBit) != 0; }

  std::optional<ObjectRef> getSubmoduleNameRef() const {
    if (isSubmodule())
      return getReference(getNumReferences() - 1);
    return std::nullopt;
  }

  /// If \c getBaseFile() is a modular header, get its submodule name.
  Expected<std::optional<StringRef>> getSubmoduleName() {
    auto Ref = getSubmoduleNameRef();
    if (!Ref)
      return std::nullopt;
    auto Node = getCAS().getProxy(*Ref);
    if (!Node)
      return Node.takeError();
    return Node->getData();
  }

  size_t getNumIncludes() const {
    return getNumReferences() - (isSubmodule() ? 2 : 1);
  };

  ObjectRef getIncludeRef(size_t I) const {
    assert(I < getNumIncludes());
    return getReference(I + 1);
  }

  enum class NodeKind : uint8_t {
    Tree,
    ModuleImport,
  };

  /// The kind of node included at the given index.
  NodeKind getIncludeKind(size_t I) const;

  /// The sub-include-tree or module import for the given index.
  Expected<Node> getIncludeNode(size_t I);

  /// The sub-include-trees of included files, in the order that they occurred.
  Expected<IncludeTree> getIncludeTree(size_t I) {
    assert(getIncludeKind(I) == NodeKind::Tree);
    return getIncludeTree(getIncludeRef(I));
  }

  /// The source byte offset for a particular include, pointing to the beginning
  /// of the line just after the #include directive. The offset represents the
  /// location at which point the include-tree would have been processed by the
  /// preprocessor and parser.
  ///
  /// For example:
  /// \code
  ///   #include "a.h" -> include-tree("a.h")
  ///   | <- include-tree-offset("a.h")
  /// \endcode
  ///
  /// Using the "after #include" offset makes it trivial to identify the part
  /// of the source file that encompasses a set of include-trees (for example in
  /// the case where want to identify the "includes preamble" of the main file.
  uint32_t getIncludeOffset(size_t I) const;

  /// The \p __has_include() preprocessor checks, in the order that they
  /// occurred. The source offsets for the checks are not tracked, "replaying"
  /// the include-tree depends on the invariant that the same exact checks will
  /// occur in the same order.
  bool getCheckResult(size_t I) const;

  /// Passes pairs of (Node, include offset) to \p Callback.
  llvm::Error forEachInclude(
      llvm::function_ref<llvm::Error(std::pair<Node, uint32_t>)> Callback);

  struct IncludeInfo {
    ObjectRef Ref; ///< IncludeTree or IncludeTreeModuleImport
    uint32_t Offset;
    NodeKind Kind;
  };

  static Expected<IncludeTree>
  create(ObjectStore &DB, SrcMgr::CharacteristicKind FileCharacteristic,
         ObjectRef BaseFile, ArrayRef<IncludeInfo> Includes,
         Optional<ObjectRef> SubmoduleName, llvm::SmallBitVector Checks);

  static Expected<IncludeTree> get(ObjectStore &DB, ObjectRef Ref);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

private:
  friend class IncludeTreeBase<IncludeTree>;
  friend class IncludeTreeRoot;

  static constexpr char IsSubmoduleBit = 0x80;

  explicit IncludeTree(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  Expected<IncludeTree> getIncludeTree(ObjectRef Ref) {
    auto Node = getCAS().getProxy(Ref);
    if (!Node)
      return Node.takeError();
    return IncludeTree(std::move(*Node));
  }

  Expected<Node> getIncludeNode(ObjectRef Ref, NodeKind Kind);

  StringRef dataSkippingFlags() const { return getData().drop_front(); }
  StringRef dataSkippingIncludes() const {
    return dataSkippingFlags().drop_front(getNumIncludes() *
                                          (sizeof(uint32_t) + 1));
  }

  static bool isValid(const ObjectProxy &Node);
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }
};

/// Represents a \p SourceManager file (or buffer in the case of preprocessor
/// predefines) that got included by the preprocessor.
class IncludeTree::File : public IncludeTreeBase<File> {
public:
  static constexpr StringRef getNodeKind() { return "File"; }

  ObjectRef getFilenameRef() const { return getReference(0); }
  ObjectRef getContentsRef() const { return getReference(1); }

  Expected<ObjectProxy> getFilename() {
    return getCAS().getProxy(getFilenameRef());
  }

  Expected<ObjectProxy> getContents() {
    return getCAS().getProxy(getContentsRef());
  }

  Expected<FileInfo> getFileInfo() {
    auto Filename = getFilename();
    if (!Filename)
      return Filename.takeError();
    auto Contents = getContents();
    if (!Contents)
      return Contents.takeError();
    return IncludeTree::FileInfo{Filename->getData(), Contents->getData()};
  }

  Expected<std::unique_ptr<llvm::MemoryBuffer>> getMemoryBuffer() {
    auto Filename = getFilename();
    if (!Filename)
      return Filename.takeError();
    auto Contents = getContents();
    if (!Contents)
      return Contents.takeError();
    return Contents->getMemoryBuffer(Filename->getData());
  }

  static Expected<File> create(ObjectStore &DB, StringRef Filename,
                               ObjectRef Contents);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getNumReferences() == 2 && Base.getData().empty();
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  friend class IncludeTreeBase<File>;
  friend class FileList;
  friend class IncludeTree;
  friend class IncludeTreeRoot;

  explicit File(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }
};

/// A list of \c File entries. Multiple \c FileList can be combined without
/// copying their contents. This is used along with a simple implementation of a
/// \p vfs::FileSystem produced via \p createIncludeTreeFileSystem().
class IncludeTree::FileList : public IncludeTreeBase<FileList> {
public:
  static constexpr StringRef getNodeKind() { return "List"; }

  using FileSizeTy = uint32_t;

  /// \returns each \c File entry along with its file size.
  llvm::Error
  forEachFile(llvm::function_ref<llvm::Error(File, FileSizeTy)> Callback);

  /// We record the file size as well to avoid needing to materialize the
  /// underlying buffer for the \p IncludeTreeFileSystem::status()
  /// implementation to provide the file size.
  struct FileEntry {
    ObjectRef FileRef;
    FileSizeTy Size;
  };
  static Expected<FileList> create(ObjectStore &DB, ArrayRef<FileEntry> Files,
                                   ArrayRef<ObjectRef> FileLists);

  static Expected<FileList> get(ObjectStore &CAS, ObjectRef Ref);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

private:
  friend class IncludeTreeBase<FileList>;
  friend class IncludeTreeRoot;

  explicit FileList(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  size_t getNumFilesCurrentList() const;
  FileSizeTy getFileSize(size_t I) const;

  Expected<File> getFile(ObjectRef Ref) {
    auto Node = getCAS().getProxy(Ref);
    if (!Node)
      return Node.takeError();
    return File(std::move(*Node));
  }

  llvm::Error
  forEachFileImpl(llvm::DenseSet<ObjectRef> &Seen,
                  llvm::function_ref<llvm::Error(File, FileSizeTy)> Callback);

  static bool isValid(const ObjectProxy &Node);
  static bool isValid(ObjectStore &CAS, ObjectRef Ref) {
    auto Node = CAS.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }
};

/// Represents a module imported by an IncludeTree.
class IncludeTree::ModuleImport : public IncludeTreeBase<ModuleImport> {
public:
  static constexpr StringRef getNodeKind() { return "ModI"; }

  static Expected<ModuleImport> create(ObjectStore &DB, StringRef ModuleName,
                                       bool VisibilityOnly);

  StringRef getModuleName() const { return getData().drop_front(); }
  /// Whether this module should only be "marked visible" rather than imported.
  bool visibilityOnly() const { return (bool)getData()[0]; }

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getNumReferences() == 0 && Base.getData().size() > 1;
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  friend class IncludeTreeBase<ModuleImport>;
  friend class Node;

  explicit ModuleImport(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }
};

/// Represents an \c IncludeTree or \c ModuleImport.
class IncludeTree::Node {
public:
  IncludeTree getIncludeTree() const {
    assert(K == NodeKind::Tree);
    return IncludeTree(N);
  }
  ModuleImport getModuleImport() const {
    assert(K == NodeKind::ModuleImport);
    return ModuleImport(N);
  }
  NodeKind getKind() const { return K; }

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

private:
  friend class IncludeTree;
  Node(ObjectProxy N, NodeKind K) : N(std::move(N)), K(K) {}
  ObjectProxy N;
  NodeKind K;
};

/// Module or submodule declaration as part of the \c IncludeTreeRoot module
/// map structure.
class IncludeTree::Module : public IncludeTreeBase<Module> {
public:
  static constexpr StringRef getNodeKind() { return "Modu"; }

  class ExportList;
  class LinkLibraryList;

  struct ModuleFlags {
    bool IsFramework : 1;
    bool IsExplicit : 1;
    bool IsExternC : 1;
    bool IsSystem : 1;
    bool InferSubmodules : 1;
    bool InferExplicitSubmodules : 1;
    bool InferExportWildcard : 1;
    ModuleFlags()
        : IsFramework(false), IsExplicit(false), IsExternC(false),
          IsSystem(false), InferSubmodules(false),
          InferExplicitSubmodules(false), InferExportWildcard(false) {}
  };

  ModuleFlags getFlags() const;

  /// The name of the current (sub)module.
  StringRef getName() const { return dataAfterFlags(); }

  size_t getNumSubmodules() const;

  Expected<Module> getSubmodule(size_t I) {
    assert(I < getNumSubmodules());
    auto Node = getCAS().getProxy(getReference(I));
    if (!Node)
      return Node.takeError();
    return Module(*Node);
  }

  llvm::Error forEachSubmodule(llvm::function_ref<llvm::Error(Module)> CB);

  std::optional<ObjectRef> getExportsRef() const {
    if (std::optional<unsigned> Index = getExportsIndex())
      return getReference(*Index);
    return std::nullopt;
  }
  std::optional<ObjectRef> getLinkLibrariesRef() const {
    if (std::optional<unsigned> Index = getLinkLibrariesIndex())
      return getReference(*Index);
    return std::nullopt;
  }

  /// The list of modules that this submodule re-exports.
  Expected<std::optional<ExportList>> getExports();

  /// The list of modules that this submodule re-exports.
  Expected<std::optional<LinkLibraryList>> getLinkLibraries();

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static Expected<Module> create(ObjectStore &DB, StringRef ModuleName,
                                 ModuleFlags Flags,
                                 ArrayRef<ObjectRef> Submodules,
                                 std::optional<ObjectRef> ExportList,
                                 std::optional<ObjectRef> LinkLibraries);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getData().size() > 2;
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  uint16_t rawFlags() const;
  StringRef dataAfterFlags() const { return getData().drop_front(2); }
  bool hasExports() const;
  bool hasLinkLibraries() const;
  std::optional<unsigned> getExportsIndex() const;
  std::optional<unsigned> getLinkLibrariesIndex() const;

  explicit Module(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  friend class IncludeTreeBase<Module>;
  friend class IncludeTreeRoot;
  friend class ModuleMap;
};

/// The set of modules re-exported by another module.
class IncludeTree::Module::ExportList : public IncludeTreeBase<ExportList> {
public:
  static constexpr StringRef getNodeKind() { return "ExpL"; }

  /// An explicit export.
  struct Export {
    StringRef ModuleName;
    bool Wildcard;
  };

  /// Whether this module exports all imported modules (`export *`).
  bool hasGlobalWildcard() const;

  size_t getNumExplicitExports() const { return getNumReferences(); }

  /// Whether the explicit export at \p I has a wildcard (`export MyModule.*`).
  bool exportHasWildcard(size_t I) const;

  Expected<Export> getExplicitExport(size_t I);

  /// Calls \p CB for each explicit export declaration.
  llvm::Error forEachExplicitExport(llvm::function_ref<llvm::Error(Export)> CB);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static Expected<ExportList> create(ObjectStore &DB, ArrayRef<Export> Exports,
                                     bool GlobalWildcard);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    size_t ExpectedBitSize = Base.getNumReferences() + 1;
    size_t ExpectedSize =
        ExpectedBitSize / 8 + (ExpectedBitSize % 8 == 0 ? 0 : 1);
    return Base.getData().size() == ExpectedSize;
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  explicit ExportList(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  friend class IncludeTreeBase<ExportList>;
  friend class Module;
};

/// The set of libraries to link against when using a module.
class IncludeTree::Module::LinkLibraryList
    : public IncludeTreeBase<LinkLibraryList> {
public:
  static constexpr StringRef getNodeKind() { return "LnkL"; }

  struct LinkLibrary {
    StringRef Library;
    bool IsFramework;
  };

  size_t getNumLibraries() const { return getNumReferences(); }

  /// Whether the library at \p I is a framework.
  bool isFramework(size_t I) const;

  ObjectRef getLibraryNameRef(size_t I) const { return getReference(I); }

  Expected<LinkLibrary> getLinkLibrary(size_t I) {
    auto Name = getCAS().getProxy(getLibraryNameRef(I));
    if (!Name)
      return Name.takeError();
    return LinkLibrary{Name->getData(), isFramework(I)};
  }

  /// Calls \p CB for each link libary.
  llvm::Error
  forEachLinkLibrary(llvm::function_ref<llvm::Error(LinkLibrary)> CB);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static Expected<LinkLibraryList> create(ObjectStore &DB,
                                          ArrayRef<LinkLibrary> Exports);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    size_t ExpectedBitSize = Base.getNumReferences();
    size_t ExpectedSize =
        ExpectedBitSize / 8 + (ExpectedBitSize % 8 == 0 ? 0 : 1);
    return Base.getData().size() == ExpectedSize;
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  explicit LinkLibraryList(ObjectProxy Node)
      : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  friend class IncludeTreeBase<LinkLibraryList>;
  friend class Module;
};

class IncludeTree::ModuleMap : public IncludeTreeBase<ModuleMap> {
public:
  static constexpr StringRef getNodeKind() { return "ModM"; }

  size_t getNumModules() const { return getNumReferences(); }

  Expected<Module> getModule(size_t I) {
    auto Node = getCAS().getProxy(getReference(I));
    if (!Node)
      return Node.takeError();
    return Module(*Node);
  }

  /// Calls \p CB for each module declaration.
  llvm::Error forEachModule(llvm::function_ref<llvm::Error(Module)> CB);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static Expected<ModuleMap> create(ObjectStore &DB,
                                    ArrayRef<ObjectRef> Modules);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getData().empty();
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  explicit ModuleMap(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  friend class IncludeTreeBase<ModuleMap>;
  friend class IncludeTreeRoot;
};

/// Represents the include-tree result for a translation unit.
class IncludeTreeRoot : public IncludeTreeBase<IncludeTreeRoot> {
public:
  static constexpr StringRef getNodeKind() { return "Root"; }

  ObjectRef getMainFileTreeRef() const { return getReference(0); }

  ObjectRef getFileListRef() const { return getReference(1); }

  Optional<ObjectRef> getPCHRef() const {
    if (auto Index = getPCHRefIndex())
      return getReference(*Index);
    return std::nullopt;
  }

  Optional<ObjectRef> getModuleMapRef() const {
    if (auto Index = getModuleMapRefIndex())
      return getReference(*Index);
    return std::nullopt;
  }

  Expected<IncludeTree> getMainFileTree() {
    auto Node = getCAS().getProxy(getMainFileTreeRef());
    if (!Node)
      return Node.takeError();
    return IncludeTree(std::move(*Node));
  }

  Expected<IncludeTree::FileList> getFileList() {
    auto Node = getCAS().getProxy(getFileListRef());
    if (!Node)
      return Node.takeError();
    return IncludeTree::FileList(std::move(*Node));
  }

  Expected<Optional<StringRef>> getPCHBuffer() {
    if (Optional<ObjectRef> Ref = getPCHRef()) {
      auto Node = getCAS().getProxy(*Ref);
      if (!Node)
        return Node.takeError();
      return Node->getData();
    }
    return None;
  }

  Expected<Optional<IncludeTree::ModuleMap>> getModuleMap() {
    if (Optional<ObjectRef> Ref = getModuleMapRef()) {
      auto Node = getCAS().getProxy(*Ref);
      if (!Node)
        return Node.takeError();
      return IncludeTree::ModuleMap(*Node);
    }
    return std::nullopt;
  }

  static Expected<IncludeTreeRoot>
  create(ObjectStore &DB, ObjectRef MainFileTree, ObjectRef FileList,
         Optional<ObjectRef> PCHRef,
         Optional<ObjectRef> ModuleMapRef);

  static Expected<IncludeTreeRoot> get(ObjectStore &DB, ObjectRef Ref);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return (Base.getNumReferences() >= 2 && Base.getNumReferences() <= 4) &&
           Base.getData().size() == 1;
  }
  static bool isValid(ObjectStore &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  friend class IncludeTreeBase<IncludeTreeRoot>;

  std::optional<unsigned> getPCHRefIndex() const;
  std::optional<unsigned> getModuleMapRefIndex() const;

  explicit IncludeTreeRoot(ObjectProxy Node)
      : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }
};

/// An implementation of a \p vfs::FileSystem that supports the simple queries
/// of the preprocessor, for creating \p FileEntries using a file path, while
/// "replaying" an \p IncludeTreeRoot. It is not intended to be a complete
/// implementation of a file system.
Expected<IntrusiveRefCntPtr<llvm::vfs::FileSystem>>
createIncludeTreeFileSystem(IncludeTreeRoot &Root);

} // namespace cas
} // namespace clang

#endif
