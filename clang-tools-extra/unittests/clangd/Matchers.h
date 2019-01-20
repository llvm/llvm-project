//===-- Matchers.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GMock matchers that aren't specific to particular tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_MATCHERS_H
#include "Protocol.h"
#include "gmock/gmock.h"

namespace clang {
namespace clangd {
using ::testing::Matcher;

// EXPECT_IFF expects matcher if condition is true, and Not(matcher) if false.
// This is hard to write as a function, because matchers may be polymorphic.
#define EXPECT_IFF(condition, value, matcher)                                  \
  do {                                                                         \
    if (condition)                                                             \
      EXPECT_THAT(value, matcher);                                             \
    else                                                                       \
      EXPECT_THAT(value, ::testing::Not(matcher));                             \
  } while (0)

// HasSubsequence(m1, m2, ...) matches a vector containing elements that match
// m1, m2 ... in that order.
//
// SubsequenceMatcher implements this once the type of vector is known.
template <typename T>
class SubsequenceMatcher
    : public ::testing::MatcherInterface<const std::vector<T> &> {
  std::vector<Matcher<T>> Matchers;

public:
  SubsequenceMatcher(std::vector<Matcher<T>> M) : Matchers(M) {}

  void DescribeTo(std::ostream *OS) const override {
    *OS << "Contains the subsequence [";
    const char *Sep = "";
    for (const auto &M : Matchers) {
      *OS << Sep;
      M.DescribeTo(OS);
      Sep = ", ";
    }
    *OS << "]";
  }

  bool MatchAndExplain(const std::vector<T> &V,
                       ::testing::MatchResultListener *L) const override {
    std::vector<int> Matches(Matchers.size());
    size_t I = 0;
    for (size_t J = 0; I < Matchers.size() && J < V.size(); ++J)
      if (Matchers[I].Matches(V[J]))
        Matches[I++] = J;
    if (I == Matchers.size()) // We exhausted all matchers.
      return true;
    if (L->IsInterested()) {
      *L << "\n  Matched:";
      for (size_t K = 0; K < I; ++K) {
        *L << "\n\t";
        Matchers[K].DescribeTo(L->stream());
        *L << " ==> " << ::testing::PrintToString(V[Matches[K]]);
      }
      *L << "\n\t";
      Matchers[I].DescribeTo(L->stream());
      *L << " ==> no subsequent match";
    }
    return false;
  }
};

// PolySubsequenceMatcher implements a "polymorphic" SubsequenceMatcher.
// It captures the types of the element matchers, and can be converted to
// Matcher<vector<T>> if each matcher can be converted to Matcher<T>.
// This allows HasSubsequence() to accept polymorphic matchers like Not().
template <typename... M> class PolySubsequenceMatcher {
  std::tuple<M...> Matchers;

public:
  PolySubsequenceMatcher(M &&... Args)
      : Matchers(std::make_tuple(std::forward<M>(Args)...)) {}

  template <typename T> operator Matcher<const std::vector<T> &>() const {
    return ::testing::MakeMatcher(new SubsequenceMatcher<T>(
        TypedMatchers<T>(llvm::index_sequence_for<M...>{})));
  }

private:
  template <typename T, size_t... I>
  std::vector<Matcher<T>> TypedMatchers(llvm::index_sequence<I...>) const {
    return {std::get<I>(Matchers)...};
  }
};

// HasSubsequence(m1, m2, ...) matches a vector containing elements that match
// m1, m2 ... in that order.
// The real implementation is in SubsequenceMatcher.
template <typename... Args>
PolySubsequenceMatcher<Args...> HasSubsequence(Args &&... M) {
  return PolySubsequenceMatcher<Args...>(std::forward<Args>(M)...);
}

// EXPECT_ERROR seems like a pretty generic name, make sure it's not defined
// already.
#ifdef EXPECT_ERROR
#error "Refusing to redefine EXPECT_ERROR"
#endif

// Consumes llvm::Expected<T>, checks it contains an error and marks it as
// handled.
#define EXPECT_ERROR(expectedValue)                                            \
  do {                                                                         \
    auto &&ComputedValue = (expectedValue);                                    \
    if (ComputedValue) {                                                       \
      ADD_FAILURE() << "expected an error from " << #expectedValue             \
                    << " but got "                                             \
                    << ::testing::PrintToString(*ComputedValue);               \
      break;                                                                   \
    }                                                                          \
    llvm::consumeError(ComputedValue.takeError());                             \
  } while (false)

} // namespace clangd
} // namespace clang
#endif
