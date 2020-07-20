// RUN: %check_clang_tidy %s cppcoreguidelines-const-correctness %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: "cppcoreguidelines-const-correctness.AnalyzeValues", value: 1},\
// RUN:   {key: "cppcoreguidelines-const-correctness.WarnPointersAsValues", value: 1}, \
// RUN:   {key: "cppcoreguidelines-const-correctness.TransformPointersAsValues", value: 1},\
// RUN:  ]}' --

void potential_const_pointer() {
  double np_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double *p_local0 = &np_local0[1];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double *' can be declared 'const'
  // CHECK-FIXES: const
}
