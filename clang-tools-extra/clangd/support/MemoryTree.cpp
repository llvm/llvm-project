#include "support/MemoryTree.h"
#include "Trace.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>

namespace clang {
namespace clangd {

namespace {

size_t traverseTree(const MemoryTree &MT, std::string &ComponentName,
                    const trace::Metric &Out) {
  size_t OriginalLen = ComponentName.size();
  if (!ComponentName.empty())
    ComponentName += '.';
  size_t Total = MT.self();
  for (const auto &Entry : MT.children()) {
    ComponentName += Entry.first;
    Total += traverseTree(Entry.getSecond(), ComponentName, Out);
    ComponentName.resize(OriginalLen + 1);
  }
  ComponentName.resize(OriginalLen);
  Out.record(Total, ComponentName);
  return Total;
}
} // namespace

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

void record(const MemoryTree &MT, std::string RootName,
            const trace::Metric &Out) {
  traverseTree(MT, RootName, Out);
}
} // namespace clangd
} // namespace clang
