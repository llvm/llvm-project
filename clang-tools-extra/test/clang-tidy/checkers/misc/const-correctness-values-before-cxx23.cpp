// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:     misc-const-correctness.TransformValues: true, \
// RUN:     misc-const-correctness.WarnPointersAsValues: false, \
// RUN:     misc-const-correctness.TransformPointersAsValues: false \
// RUN:   }}" -- -fno-delayed-template-parsing


double &non_const_ref_return() {
  double p_local0 = 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double' can be declared 'const'
  // CHECK-FIXES: double const p_local0
  double np_local0 = 42.42;
  return np_local0;
}

double *&return_non_const_pointer_ref() {
  double *np_local0 = nullptr;
  return np_local0;
}
