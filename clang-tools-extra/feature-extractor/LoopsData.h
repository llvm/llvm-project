#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <clang/AST/Stmt.h>

#include "NaryTree.h"

using namespace clang;

class LoopsData {
public:
  struct MetaData {
    const Stmt *for_stmt;
    llvm::APInt loop_range;
    std::size_t float_ops;
    std::size_t int_ops;
    std::size_t mem_loads;
    std::size_t mem_stores;

    MetaData(const Stmt *fs) : for_stmt(fs) {}
    MetaData(const Stmt *fs, const llvm::APInt &rng, std::size_t float_ops,
             std::size_t int_ops, std::size_t lds, std::size_t strs)
        : for_stmt(fs), loop_range(rng), float_ops(float_ops), int_ops(int_ops),
          mem_loads(lds), mem_stores(strs) {}

    friend bool operator==(const MetaData &lhs, const MetaData &rhs) {
      return lhs.for_stmt == rhs.for_stmt;
    }
  };

  using TreeType = NaryTree<MetaData>;

  void add_for(clang::ASTContext *context, const Stmt *parent,
               const MetaData &self) {
    ids[self.for_stmt] = self.for_stmt->getID(*context);

    if (parent == nullptr) {
      loops.push_back({});
      loops.back().add_node(nullptr, self);
    } else {
      auto ntree = std::find_if(
          loops.begin(), loops.end(),
          [&parent](const TreeType &tree) { return tree.contains(parent); });

      if (ntree != loops.end()) {
        ntree->add_node(parent, self);
      }
    }
  }

  auto &get_ids() { return ids; }
  auto &get_loops() { return loops; }

private:
  std::unordered_map<const Stmt *, std::int64_t> ids;
  std::vector<TreeType> loops;
};
