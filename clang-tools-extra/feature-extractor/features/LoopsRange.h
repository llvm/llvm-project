#pragma once

#include <array>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "../LoopsData.h"
#include "../utils.h"
#include "../visitors/FloatOpCounter.h"
#include "../visitors/IntegerOpCounter.h"
#include "../visitors/MemoryAccessCounter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;

class LoopsRange : public MatchFinder::MatchCallback {
  LoopsData loops_data;

public:
  static inline std::array Matchers = {
      forStmt(unless(hasAncestor(forStmt()))).bind("topLevelFor"),
      forStmt(hasAncestor(forStmt())).bind("nestedFor"),
      // forStmt(hasParent(compoundStmt(hasParent(forStmt())))).bind("nestedFor"),
  };

  virtual void run(const MatchFinder::MatchResult &result) override {
    static constexpr auto GatherData =
        [](const MatchFinder::MatchResult &result, LoopsData &loops_data,
           const clang::ForStmt *parent_for, const clang::ForStmt *fs) {
          // llvm::outs() << "Nested for loop at ";
          // fs->getForLoc().print(llvm::outs(), *result.SourceManager);
          // llvm::outs() << "\n";

          FloatOpCounter fCounter;
          IntegerOpCounter iCounter;
          MemoryAccessCounter memCounter;

          fCounter.traverse(const_cast<Stmt *>(fs->getBody()));
          iCounter.traverse(const_cast<Stmt *>(fs->getBody()));
          memCounter.traverse(const_cast<Stmt *>(fs->getBody()));

          loops_data.add_for(
              result.Context, parent_for,
              LoopsData::MetaData{
                  fs, Utils::get_total_for_repetition_count(result.Context, fs),
                  fCounter.get_count(), iCounter.get_count(),
                  memCounter.get_load_count(), memCounter.get_store_count()});
        };

    if (const ForStmt *fs = result.Nodes.getNodeAs<ForStmt>("topLevelFor");
        Utils::is_in_main_file(result.Context, fs)) {
      GatherData(result, loops_data, nullptr, fs);
    }

    if (const ForStmt *fs = result.Nodes.getNodeAs<ForStmt>("nestedFor");
        Utils::is_in_main_file(result.Context, fs)) {
      if (auto parent_for =
              Utils::get_parent_stmt<ForStmt>(result.Context, fs)) {
        GatherData(result, loops_data, parent_for, fs);
      }
    }
  }

  static const char *get_title() { return "loops_range"; }
  std::size_t get_result() {
    llvm::outs() << "\n";

    for (auto &loop : loops_data.get_loops()) {
      loop.traverse_pre_order(
          [](const LoopsData::TreeType::TraverseResult &result) mutable {
            const auto &[optParentStmt, selfMetaData, depth, isLeaf] = result;

            llvm::outs() << std::string(depth * 2, ' ') << "for "
                         << (isLeaf ? "(leaf) " : "")
                         << "loop range: " << selfMetaData.loop_range
                         << ", float ops: " << selfMetaData.float_ops
                         << ", int ops: " << selfMetaData.int_ops
                         << ", mem loads: " << selfMetaData.mem_loads
                         << ", mem stores: " << selfMetaData.mem_stores << "\n";
          });
    }
    return loops_data.get_ids().size();
  }
};
