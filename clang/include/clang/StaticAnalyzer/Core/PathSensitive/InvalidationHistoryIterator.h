//===- InvalidationHistoryIterator.h -----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONHISTORYITERATOR_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONHISTORYITERATOR_H

#include <cassert>
#include <cstddef>
#include <iterator>

namespace clang::ento {
class SVal;
class SymExpr;

class InvalidationHistoryIterator {
public:
  InvalidationHistoryIterator() = default;
  explicit InvalidationHistoryIterator(const SymExpr *Sym) : Curr(Sym) {}
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = const SymExpr *;
  using reference = const SymExpr *const &;
  using pointer = const SymExpr *const *;

  InvalidationHistoryIterator &operator++();

  InvalidationHistoryIterator operator++(int) {
    auto Tmp = *this;
    ++*this;
    return Tmp;
  }

  reference operator*() const {
    assert(Curr && "Cannot dereference end iterator!");
    return Curr;
  }

  bool operator==(InvalidationHistoryIterator Other) const {
    return Curr == Other.Curr;
  }
  bool operator!=(InvalidationHistoryIterator Other) const {
    return !(*this == Other);
  }

private:
  const SymExpr *Curr = nullptr;
};

} // namespace clang::ento

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_INVALIDATIONHISTORYITERATOR_H
