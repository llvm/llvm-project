#pragma once

#include <array>
#include <cstddef>
#include <numeric>
#include <unordered_map>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;

class OmpRegions : public MatchFinder::MatchCallback {
  std::unordered_map<std::string, unsigned> regions_count;

public:
  static inline std::array Matchers = {
      ompExecutableDirective().bind("ompRegion")};

  virtual void run(const MatchFinder::MatchResult &result) override {
    if (const auto *omp_directive =
            result.Nodes.getNodeAs<OMPExecutableDirective>("ompRegion")) {
      std::string omp_type;

      using namespace llvm;
      if (isa<OMPParallelDirective>(omp_directive))
        omp_type = "parallel";
      else if (isa<OMPForDirective>(omp_directive))
        omp_type = "for";
      else if (isa<OMPParallelForDirective>(omp_directive))
        omp_type = "parallel for";
      else if (isa<OMPSingleDirective>(omp_directive))
        omp_type = "single";
      else if (isa<OMPMasterDirective>(omp_directive))
        omp_type = "master";
      else if (isa<OMPCriticalDirective>(omp_directive))
        omp_type = "critical";
      else if (isa<OMPTaskDirective>(omp_directive))
        omp_type = "task";
      else if (isa<OMPSectionDirective>(omp_directive))
        omp_type = "section";
      else if (isa<OMPSectionsDirective>(omp_directive))
        omp_type = "sections";
      else if (isa<OMPBarrierDirective>(omp_directive))
        omp_type = "barrier";
      else
        omp_type = "other";

      regions_count[omp_type]++;
    }
  }

  static const char *get_title() { return "opm_regions"; }
  std::size_t get_result() {
#ifndef NDEBUG
    llvm::outs() << "\n";
    for (const auto &pair : regions_count)
      llvm::outs() << "OMP region [" << pair.first << "]: " << pair.second
                   << "\n";
#endif

    return std::accumulate(
        regions_count.cbegin(), regions_count.cend(), std::size_t{0},
        [](std::size_t acc, const auto &p) { return acc + p.second; });
  }
};
