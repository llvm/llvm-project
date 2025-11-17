// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

#define VOID_F0 void f0
#define VOID_F1_BRACES_BODY void f1() {}

void f1();
namespace foo {
  void f0();
  void f1();
}
VOID_F0() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: free function 'f0' shadows 'foo::f0' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
VOID_F1_BRACES_BODY
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: free function 'f1' shadows 'foo::f1' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
