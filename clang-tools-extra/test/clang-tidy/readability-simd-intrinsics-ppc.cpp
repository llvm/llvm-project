// RUN: %check_clang_tidy %s readability-simd-intrinsics %t -- \
// RUN:  -config='{CheckOptions: [ \
// RUN:    {key: readability-simd-intrinsics.Suggest, value: 1} \
// RUN:  ]}' -- -target ppc64le -maltivec -std=c++11

vector int vec_add(vector int, vector int);

void PPC() {
  vector int i0, i1;

  vec_add(i0, i1);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'vec_add' can be replaced by operator+ on std::experimental::simd objects [readability-simd-intrinsics]
}
