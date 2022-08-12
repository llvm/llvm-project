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
#include "llvm/CAS/CASDB.h"

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

  static Expected<NodeT> create(CASDB &DB, ArrayRef<ObjectRef> Refs,
                                ArrayRef<char> Data);

  static bool isValid(const ObjectProxy &Node) {
    return Node.getData().startswith(NodeT::getNodeKind());
  }

  friend class IncludeFile;
  friend class IncludeTree;
  friend class IncludeTreeRoot;
};

/// Represents a \p SourceManager file (or buffer in the case of preprocessor
/// predefines) that got included by the preprocessor.
class IncludeFile : public IncludeTreeBase<IncludeFile> {
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

  struct FileInfo {
    StringRef Filename;
    StringRef Contents;
  };

  Expected<FileInfo> getFileInfo() {
    auto Filename = getFilename();
    if (!Filename)
      return Filename.takeError();
    auto Contents = getContents();
    if (!Contents)
      return Contents.takeError();
    return FileInfo{Filename->getData(), Contents->getData()};
  }

  static Expected<IncludeFile> create(CASDB &DB, StringRef Filename,
                                      ObjectRef Contents);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getNumReferences() == 2 && Base.getData().empty();
  }
  static bool isValid(CASDB &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  friend class IncludeTreeBase<IncludeFile>;
  friend class IncludeTree;
  friend class IncludeTreeRoot;

  explicit IncludeFile(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }
};

/// Represents a DAG of included files by the preprocessor.
/// Each node in the DAG represents a particular inclusion of a file that
/// encompasses inclusions of other files as sub-trees, along with all the
/// \p __has_include() preprocessor checks that occurred during preprocessing
/// of that file.
class IncludeTree : public IncludeTreeBase<IncludeTree> {
public:
  static constexpr StringRef getNodeKind() { return "Tree"; }

  Expected<IncludeFile> getBaseFile() {
    auto Node = getCAS().getProxy(getBaseFileRef());
    if (!Node)
      return Node.takeError();
    return IncludeFile(std::move(*Node));
  }

  /// The include file that resulted in this include-tree.
  ObjectRef getBaseFileRef() const { return getReference(0); }

  Expected<IncludeFile::FileInfo> getBaseFileInfo() {
    auto File = getBaseFile();
    if (!File)
      return File.takeError();
    return File->getFileInfo();
  }

  SrcMgr::CharacteristicKind getFileCharacteristic() const {
    return (SrcMgr::CharacteristicKind)dataSkippingIncludes().front();
  }

  size_t getNumIncludes() const { return getNumReferences() - 1; }

  ObjectRef getIncludeRef(size_t I) const {
    assert(I < getNumIncludes());
    return getReference(I + 1);
  }

  /// The sub-include-trees of included files, in the order that they occurred.
  Expected<IncludeTree> getInclude(size_t I) {
    return getInclude(getIncludeRef(I));
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

  /// Passes pairs of (IncludeTree, include offset) to \p Callback.
  llvm::Error forEachInclude(
      llvm::function_ref<llvm::Error(std::pair<IncludeTree, uint32_t>)>
          Callback);

  static Expected<IncludeTree>
  create(CASDB &DB, SrcMgr::CharacteristicKind FileCharacteristic,
         ObjectRef BaseFile, ArrayRef<std::pair<ObjectRef, uint32_t>> Includes,
         llvm::SmallBitVector Checks);

  static Expected<IncludeTree> get(CASDB &DB, ObjectRef Ref);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

private:
  friend class IncludeTreeBase<IncludeTree>;
  friend class IncludeTreeRoot;

  explicit IncludeTree(ObjectProxy Node) : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }

  Expected<IncludeTree> getInclude(ObjectRef Ref) {
    auto Node = getCAS().getProxy(Ref);
    if (!Node)
      return Node.takeError();
    return IncludeTree(std::move(*Node));
  }

  StringRef dataSkippingIncludes() const {
    return getData().drop_front(getNumIncludes() * sizeof(uint32_t));
  }

  static bool isValid(const ObjectProxy &Node);
  static bool isValid(CASDB &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }
};

/// Represents the include-tree result for a translation unit.
class IncludeTreeRoot : public IncludeTreeBase<IncludeTreeRoot> {
public:
  static constexpr StringRef getNodeKind() { return "Root"; }

  ObjectRef getMainFileTreeRef() const { return getReference(0); }

  Expected<IncludeTree> getMainFileTree() {
    auto Node = getCAS().getProxy(getMainFileTreeRef());
    if (!Node)
      return Node.takeError();
    return IncludeTree(std::move(*Node));
  }

  static Expected<IncludeTreeRoot> create(CASDB &DB, ObjectRef MainFileTree);

  static Expected<IncludeTreeRoot> get(CASDB &DB, ObjectRef Ref);

  llvm::Error print(llvm::raw_ostream &OS, unsigned Indent = 0);

  static bool isValid(const ObjectProxy &Node) {
    if (!IncludeTreeBase::isValid(Node))
      return false;
    IncludeTreeBase Base(Node);
    return Base.getNumReferences() == 1 && Base.getData().empty();
  }
  static bool isValid(CASDB &DB, ObjectRef Ref) {
    auto Node = DB.getProxy(Ref);
    if (!Node) {
      llvm::consumeError(Node.takeError());
      return false;
    }
    return isValid(*Node);
  }

private:
  friend class IncludeTreeBase<IncludeTreeRoot>;

  explicit IncludeTreeRoot(ObjectProxy Node)
      : IncludeTreeBase(std::move(Node)) {
    assert(isValid(*this));
  }
};

} // namespace cas
} // namespace clang

#endif
