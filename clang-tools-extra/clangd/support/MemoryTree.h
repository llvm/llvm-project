//===--- MemoryTree.h - A special tree for components and sizes -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_MEMORYTREE_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_MEMORYTREE_H_

#include "Trace.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include <cstddef>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// A tree that can be used to represent memory usage of nested components while
/// preserving the hierarchy.
/// Edges have associated names. An edge that might not be interesting to all
/// traversers or costly to copy (e.g. file names) can be marked as "detail".
/// Tree construction allows chosing between a detailed and brief mode, in brief
/// mode all "detail" edges are ignored and tree is constructed without any
/// string copies.
struct MemoryTree {
public:
  /// If Alloc is nullptr, tree is in brief mode and will ignore detail edges.
  MemoryTree(llvm::BumpPtrAllocator *DetailAlloc = nullptr)
      : DetailAlloc(DetailAlloc) {}

  /// No copy of the \p Name.
  /// Note that returned pointers are invalidated with subsequent calls to
  /// child/detail.
  MemoryTree &child(llvm::StringLiteral Name) { return createChild(Name); }

  MemoryTree(const MemoryTree &) = delete;
  MemoryTree &operator=(const MemoryTree &) = delete;

  MemoryTree(MemoryTree &&) = default;
  MemoryTree &operator=(MemoryTree &&) = default;

  /// Makes a copy of the \p Name in detailed mode, returns current node
  /// otherwise.
  /// Note that returned pointers are invalidated with subsequent calls to
  /// child/detail.
  MemoryTree &detail(llvm::StringRef Name) {
    return DetailAlloc ? createChild(Name.copy(*DetailAlloc)) : *this;
  }

  /// Increases size of current node by \p Increment.
  void addUsage(size_t Increment) { Size += Increment; }

  /// Returns edges to direct children of this node.
  const llvm::DenseMap<llvm::StringRef, MemoryTree> &children() const;

  /// Returns total number of bytes used by this sub-tree. Performs a traversal.
  size_t total() const;

  /// Returns total number of bytes used by this node only.
  size_t self() const { return Size; }

private:
  /// Adds a child with an edge labeled as \p Name. Multiple calls to this
  /// function returns the same node.
  MemoryTree &createChild(llvm::StringRef Name);

  /// Allocator to use for detailed edge names.
  llvm::BumpPtrAllocator *DetailAlloc = nullptr;

  /// Bytes owned by this component specifically.
  size_t Size = 0;

  /// Edges from current node to its children. Keys are the labels for edges.
  llvm::DenseMap<llvm::StringRef, MemoryTree> Children;
};

/// Records total memory usage of each node under \p Out. Labels are edges on
/// the path joined with ".", starting with \p RootName.
void record(const MemoryTree &MT, std::string RootName,
            const trace::Metric &Out);

} // namespace clangd
} // namespace clang

#endif
