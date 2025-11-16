// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

void f1();
namespace foo {
  void f0();
  void f1();
}
void f0() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0' shadows 'foo::f0' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo::f0() {}
void f1() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1' shadows 'foo::f1' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
