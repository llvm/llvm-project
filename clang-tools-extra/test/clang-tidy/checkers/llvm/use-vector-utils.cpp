// RUN: %check_clang_tidy %s llvm-use-vector-utils %t

// CHECK-FIXES: #include "llvm/ADT/SmallVectorExtras.h"

namespace llvm {

template <typename T> class SmallVector {};

template <typename RangeT>
SmallVector<int> to_vector(RangeT &&Range);

template <unsigned Size, typename RangeT>
SmallVector<int> to_vector(RangeT &&Range);

template <typename Out, typename RangeT>
SmallVector<Out> to_vector_of(RangeT &&Range);

template <typename Out, unsigned Size, typename RangeT>
SmallVector<Out> to_vector_of(RangeT &&Range);

template <typename ContainerT, typename FuncT>
struct mapped_range {};

template <typename ContainerT, typename FuncT>
mapped_range<ContainerT, FuncT> map_range(ContainerT &&C, FuncT &&F);

// Hypothetical 3-arg overload (for future-proofing).
template <typename ContainerT, typename FuncT, typename ExtraT>
mapped_range<ContainerT, FuncT> map_range(ContainerT &&C, FuncT &&F, ExtraT &&E);

template <typename ContainerT, typename PredT>
struct filter_range {};

template <typename ContainerT, typename PredT>
filter_range<ContainerT, PredT> make_filter_range(ContainerT &&C, PredT &&P);

// Hypothetical 3-arg overload (for future-proofing).
template <typename ContainerT, typename PredT, typename ExtraT>
filter_range<ContainerT, PredT> make_filter_range(ContainerT &&C, PredT &&P, ExtraT &&E);

} // namespace llvm

int transform(int x);
bool is_even(int x);

void test_map_range() {
  llvm::SmallVector<int> vec;

  auto result = llvm::to_vector(llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result = llvm::map_to_vector(vec, transform);

  auto result_sized = llvm::to_vector<4>(llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_sized = llvm::map_to_vector<4>(vec, transform);

  auto result_global = ::llvm::to_vector(::llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_global = ::llvm::map_to_vector(vec, transform);

  // Check that comments between `to_vector(` and `map_range(` are preserved.
  auto result_comment1 = llvm::to_vector(/*keep_me*/ llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_comment1 = llvm::map_to_vector(/*keep_me*/ vec, transform);

  // Check that comments between `to_vector<9>(` and `map_range(` are preserved.
  auto result_comment2 = llvm::to_vector<9>(/*keep_me*/ llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_comment2 = llvm::map_to_vector<9>(/*keep_me*/ vec, transform);

  // Check that comments inside `map_range(` are also preserved.
  auto result_comment3 = llvm::to_vector(llvm::map_range(/*keep_me*/ vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_comment3 = llvm::map_to_vector(/*keep_me*/ vec, transform);

  // Check that comments inside explicit template argument are preserved.
  auto result_comment4 = llvm::to_vector</*keep_me*/ 7>(llvm::map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_comment4 = llvm::map_to_vector</*keep_me*/ 7>(vec, transform);

  // Check that whitespace between callee and `(` is handled correctly.
  auto result_whitespace1 = llvm::to_vector(llvm::map_range  (vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_whitespace1 = llvm::map_to_vector(vec, transform);

  // Check that comments between callee and `(` are handled correctly.
  auto result_whitespace2 = llvm::to_vector(llvm::map_range /*weird but ok*/ (vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result_whitespace2 = llvm::map_to_vector(vec, transform);
}

void test_filter_range() {
  llvm::SmallVector<int> vec;

  auto result = llvm::to_vector(llvm::make_filter_range(vec, is_even));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'filter_to_vector'
  // CHECK-FIXES: auto result = llvm::filter_to_vector(vec, is_even);

  auto result_sized = llvm::to_vector<6>(llvm::make_filter_range(vec, is_even));
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use 'filter_to_vector'
  // CHECK-FIXES: auto result_sized = llvm::filter_to_vector<6>(vec, is_even);
}

namespace llvm {

void test_inside_llvm_namespace() {
  SmallVector<int> vec;

  // Unprefixed calls inside the `llvm` namespace should also be detected.
  auto result = to_vector(map_range(vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'map_to_vector'
  // CHECK-FIXES: auto result = map_to_vector(vec, transform);
}

} // namespace llvm

// Check that an empty macro between callee and `(` is handled.
void test_macro() {
  llvm::SmallVector<int> vec;

#define EMPTY
  // No fix-it when a macro appears between callee and `(`.
  auto result = llvm::to_vector(llvm::map_range EMPTY (vec, transform));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'map_to_vector'

#undef EMPTY
}

void test_negative() {
  llvm::SmallVector<int> vec;

  // `to_vector` without inner `map_range`/`make_filter_range` should not trigger.
  auto result1 = llvm::to_vector(vec);
  auto result2 = llvm::to_vector<4>(vec);

  // Direct use of `map_range`/`make_filter_range` without `to_vector` should not trigger.
  auto mapped = llvm::map_range(vec, transform);
  auto filtered = llvm::make_filter_range(vec, is_even);

  // `to_vector_of` variants should not trigger (no `map_to_vector_of` exists).
  auto result3 = llvm::to_vector_of<long>(llvm::map_range(vec, transform));
  auto result4 = llvm::to_vector_of<long, 4>(llvm::map_range(vec, transform));
  auto result5 = llvm::to_vector_of<long>(llvm::make_filter_range(vec, is_even));
  auto result6 = llvm::to_vector_of<long, 4>(llvm::make_filter_range(vec, is_even));

  // Hypothetical 3-arg overloads should not trigger.
  auto result7 = llvm::to_vector(llvm::map_range(vec, transform, 0));
  auto result8 = llvm::to_vector(llvm::make_filter_range(vec, is_even, 0));
}
