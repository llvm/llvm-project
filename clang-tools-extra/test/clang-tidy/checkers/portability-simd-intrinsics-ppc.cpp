// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s portability-simd-intrinsics %t -- \
// RUN:  -config='{CheckOptions: [ \
// RUN:    {key: portability-simd-intrinsics.Suggest, value: 1} \
// RUN:  ]}' -- -target ppc64le -maltivec
// FIXME: Fix the checker to work in C++20 mode.

vector int vec_add(vector int, vector int);

void PPC() {
  vector int i0, i1;

  vec_add(i0, i1);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'vec_add' can be replaced by operator+ on std::experimental::simd objects [portability-simd-intrinsics]
}
