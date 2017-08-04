void f(int i, int j = 2, int k = 5);
void f(float x, float y...);
       
void test() {
  ::
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:5 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: f(<#int i#>{#, <#int j = 2#>{#, <#int k = 5#>#}#})
  // CHECK-CC1: f(<#float x#>, <#float y, ...#>)
