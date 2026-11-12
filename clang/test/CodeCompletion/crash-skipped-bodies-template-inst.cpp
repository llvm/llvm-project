// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:24:5 %s -o - 2>&1 | FileCheck %s
template <class T>
auto make_func() {
  struct impl {
    impl* func() {
      int x;
      if (x = 10) {}
      // Check that body of this function is actually skipped.
      // CHECK-NOT: crash-skipped-bodies-template-inst.cpp:7:{{[0-9]+}}: warning: using the result of an assignment as a condition without parentheses
      return this;
    }
  };

  int x;
  if (x = 10) {}
  // Check that this function is not skipped.
  // CHECK: crash-skipped-bodies-template-inst.cpp:15:9: warning: using the result of an assignment as a condition without parentheses
  return impl();
}

void foo() {
  []() {
    make_func<int>();
    m
    // CHECK: COMPLETION: make_func : [#auto#]make_func<<#class T#>>()
  };
}
