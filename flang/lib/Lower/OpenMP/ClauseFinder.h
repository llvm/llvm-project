//===-- Lower/OpenMP/ClauseFinder.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_CLAUSEFINDER_H
#define FORTRAN_LOWER_CLAUSEFINDER_H

#include "flang/Lower/OpenMP/Clauses.h"

namespace Fortran {
namespace lower {
namespace omp {

class ClauseFinder {
  using ClauseIterator = List<Clause>::const_iterator;

public:
  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end) {
    for (ClauseIterator it = begin; it != end; ++it) {
      if (std::get_if<T>(&it->u))
        return it;
    }

    return end;
  }

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  static const T *findUniqueClause(const List<Clause> &clauses,
                                   const parser::CharBlock **source = nullptr) {
    ClauseIterator it = findClause<T>(clauses.begin(), clauses.end());
    if (it != clauses.end()) {
      if (source)
        *source = &it->source;
      return &std::get<T>(it->u);
    }
    return nullptr;
  }

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  static bool findRepeatableClause(
      const List<Clause> &clauses,
      std::function<void(const T &, const parser::CharBlock &source)>
          callbackFn) {
    bool found = false;
    ClauseIterator nextIt, endIt = clauses.end();
    for (ClauseIterator it = clauses.begin(); it != endIt; it = nextIt) {
      nextIt = findClause<T>(it, endIt);

      if (nextIt != endIt) {
        callbackFn(std::get<T>(nextIt->u), nextIt->source);
        found = true;
        ++nextIt;
      }
    }
    return found;
  }
};
} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CLAUSEFINDER_H
