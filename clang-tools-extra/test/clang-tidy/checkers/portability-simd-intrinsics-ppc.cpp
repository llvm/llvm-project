// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s portability-simd-intrinsics %t -- \
// RUN:  -config='{CheckOptions: [ \
// RUN:    {key: portability-simd-intrinsics.Suggest, value: 1} \
// RUN:  ]}' -- -target ppc64le -maltivec
// RUN: %check_clang_tidy -std=c++20-or-later %s portability-simd-intrinsics -check-suffix=CXX20 %t -- \
// RUN:  -config='{CheckOptions: [ \
// RUN:    {key: portability-simd-intrinsics.Suggest, value: 1} \
// RUN:  ]}' -- -target ppc64le -maltivec

vector int vec_add(vector int, vector int);

void PPC() {
  vector int i0, i1;

  vec_add(i0, i1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'vec_add' can be replaced by operator+ on std::experimental::simd objects [portability-simd-intrinsics]
  // CHECK-MESSAGES-CXX20: :[[@LINE-2]]:3: warning: 'vec_add' can be replaced by operator+ on std::simd objects [portability-simd-intrinsics]
}
