// RUN: %check_clang_tidy -std=c++20 -check-suffixes=,RANGES %s modernize-loop-convert %t

// RUN: %check_clang_tidy -check-suffixes=,CUSTOM,CUSTOM-NO-SYS %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:   modernize-loop-convert.MakeReverseRangeFunction: 'llvm::reverse', \
// RUN:   modernize-loop-convert.MakeReverseRangeHeader: 'llvm/ADT/STLExtras.h'}}"

// RUN: %check_clang_tidy -check-suffixes=,CUSTOM,CUSTOM-SYS %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:   modernize-loop-convert.MakeReverseRangeFunction: 'llvm::reverse', \
// RUN:   modernize-loop-convert.MakeReverseRangeHeader: '<llvm/ADT/STLExtras.h>'}}"

// RUN: %check_clang_tidy -check-suffixes=,CUSTOM,CUSTOM-NO-HEADER %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:   modernize-loop-convert.MakeReverseRangeFunction: 'llvm::reverse'}}"

// Ensure the check doesn't transform reverse loops when not in c++20 mode or
// when UseCxx20ReverseRanges has been disabled
// RUN: clang-tidy %s -checks=-*,modernize-loop-convert -- -std=c++17 | count 0

// RUN: clang-tidy %s -checks=-*,modernize-loop-convert -config="{CheckOptions: \
// RUN:     {modernize-loop-convert.UseCxx20ReverseRanges: 'false'} \
// RUN:     }" -- -std=c++20 | count 0

// Ensure we get a warning if we supply the header argument without the
// function argument.
// RUN: clang-tidy %s -checks=-*,modernize-loop-convert -config="{CheckOptions: { \
// RUN:   modernize-loop-convert.MakeReverseRangeHeader: 'llvm/ADT/STLExtras.h'}}" \
// RUN: -- -std=c++17 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-HEADER-NO-FUNC \
// RUN:       -implicit-check-not="{{warning|error}}:"

// CHECK-HEADER-NO-FUNC: warning: modernize-loop-convert: 'MakeReverseRangeHeader' is set but 'MakeReverseRangeFunction' is not, disabling reverse loop transformation

// Make sure appropiate headers are included
// CHECK-FIXES-RANGES: #include <ranges>
// CHECK-FIXES-CUSTOM-NO-SYS: #include "llvm/ADT/STLExtras.h"
// CHECK-FIXES-CUSTOM-SYS: #include <llvm/ADT/STLExtras.h>

// Make sure no header is included in this example
// CHECK-FIXES-CUSTOM-NO-HEADER-NOT: #include

namespace ADL {

template <typename T>
struct Reversable {
  using iterator = T *;
  using const_iterator = const T *;

  iterator begin();
  iterator end();
  iterator rbegin();
  iterator rend();

  const_iterator begin() const;
  const_iterator end() const;
  const_iterator rbegin() const;
  const_iterator rend() const;

  const_iterator cbegin() const;
  const_iterator cend() const;
  const_iterator crbegin() const;
  const_iterator crend() const;
};

template <typename C>
constexpr auto rbegin(C& c) -> decltype(c.rbegin()) { return c.rbegin(); }
template <typename C>
constexpr auto rend(C& c) -> decltype(c.rend()) { return c.rend(); }
template <typename C>
constexpr auto rbegin(const C& c) -> decltype(c.rbegin()) { return c.rbegin(); }
template <typename C>
constexpr auto rend(const C& c) -> decltype(c.rend()) { return c.rend(); }

template <typename C>
constexpr auto crbegin(C& c) -> decltype(c.crbegin());
template <typename C>
constexpr auto crend(C& c) -> decltype(c.crend());
template <typename C>
constexpr auto crbegin(const C& c) -> decltype(c.crbegin());
template <typename C>
constexpr auto crend(const C& c) -> decltype(c.crend());

} // namespace ADL

using ADL::Reversable;

template <typename T>
void observe(const T &);
template <typename T>
void mutate(T &);

void constContainer(const Reversable<int> &Numbers) {
  for (auto I = Numbers.rbegin(), E = Numbers.rend(); I != E; ++I) {
    observe(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   observe(Number);
  //   CHECK-FIXES-NEXT: }

  for (auto I = Numbers.crbegin(), E = Numbers.crend(); I != E; ++I) {
    observe(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   observe(Number);
  //   CHECK-FIXES-NEXT: }

  for (auto I = rbegin(Numbers), E = rend(Numbers); I != E; ++I) {
    observe(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   observe(Number);
  //   CHECK-FIXES-NEXT: }

  for (auto I = crbegin(Numbers), E = crend(Numbers); I != E; ++I) {
    observe(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   observe(Number);
  //   CHECK-FIXES-NEXT: }

  // Ensure these bad loops aren't transformed.
  for (auto I = Numbers.rbegin(), E = Numbers.end(); I != E; ++I) {
    observe(*I);
  }
  for (auto I = Numbers.begin(), E = Numbers.rend(); I != E; ++I) {
    observe(*I);
  }
}

void nonConstContainer(Reversable<int> &Numbers) {
  for (auto I = Numbers.rbegin(), E = Numbers.rend(); I != E; ++I) {
    mutate(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int & Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int & Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   mutate(Number);
  //   CHECK-FIXES-NEXT: }

  for (auto I = Numbers.crbegin(), E = Numbers.crend(); I != E; ++I) {
    observe(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   observe(Number);
  //   CHECK-FIXES-NEXT: }

  for (auto I = rbegin(Numbers), E = rend(Numbers); I != E; ++I) {
    mutate(*I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES-RANGES: for (int & Number : std::ranges::reverse_view(Numbers)) {
  // CHECK-FIXES-CUSTOM: for (int & Number : llvm::reverse(Numbers)) {
  //   CHECK-FIXES-NEXT:   mutate(Number);
  //   CHECK-FIXES-NEXT: }
}
