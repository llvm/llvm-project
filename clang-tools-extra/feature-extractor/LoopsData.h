#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <clang/AST/Stmt.h>

#include "NaryTree.h"

using namespace clang;

class LoopsData {
public:
  using TreeType = NaryTree<const Stmt *>;

  void add_for(clang::ASTContext *context, const Stmt *parent,
               const Stmt *self) {
    ids[self] = self->getID(*context);

    if (parent == nullptr) {
      loops.push_back({});
      loops.back().add_node(nullptr, self);
    } else {
      auto ntree = std::find_if(loops.begin(), loops.end(),
                                [&parent](const NaryTree<const Stmt *> &tree) {
                                  return tree.contains(parent);
                                });

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
