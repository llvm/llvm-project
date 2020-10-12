#include "support/MemoryTree.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>

namespace clang {
namespace clangd {

MemoryTree &MemoryTree::createChild(llvm::StringRef Name) {
  auto &Child = Children.try_emplace(Name, DetailAlloc).first->getSecond();
  return Child;
}

const llvm::DenseMap<llvm::StringRef, MemoryTree> &
MemoryTree::children() const {
  return Children;
}

size_t MemoryTree::total() const {
  size_t Total = Size;
  for (const auto &Entry : Children)
    Total += Entry.getSecond().total();
  return Total;
}
} // namespace clangd
} // namespace clang
