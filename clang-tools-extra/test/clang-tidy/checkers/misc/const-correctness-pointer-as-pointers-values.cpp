// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: {\
// RUN:   misc-const-correctness.AnalyzeValues: false,\
// RUN:   misc-const-correctness.AnalyzeReferences: false,\
// RUN:   misc-const-correctness.AnalyzePointers: true,\
// RUN:   misc-const-correctness.WarnPointersAsValues: true,\
// RUN:   misc-const-correctness.WarnPointersAsPointers: true,\
// RUN:   misc-const-correctness.TransformPointersAsValues: true,\
// RUN:   misc-const-correctness.TransformPointersAsPointers: true\
// RUN: }}' \
// RUN: -- -fno-delayed-template-parsing

void pointee_to_const() {
  int a[] = {1, 2};
  int *p_local0 = &a[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p_local0' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'p_local0' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int  const*const p_local0 = &a[0];
}
