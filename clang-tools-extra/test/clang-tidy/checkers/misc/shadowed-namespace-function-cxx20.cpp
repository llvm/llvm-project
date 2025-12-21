// RUN: %check_clang_tidy -std=c++20-or-later %s misc-shadowed-namespace-function %t

void f1_nested_inline_ns();
namespace foo_nested_inline_ns::inline foo2::foo3 {
  void f0_nested_inline_ns();
  void f1_nested_inline_ns();
}
void f0_nested_inline_ns() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_nested_inline_ns' shadows 'foo_nested_inline_ns::foo3::f0_nested_inline_ns' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_nested_inline_ns::foo3::f0_nested_inline_ns() {}
void f1_nested_inline_ns() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_nested_inline_ns' shadows 'foo_nested_inline_ns::foo3::f1_nested_inline_ns' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

void f1_nested_inline_ns_2();
namespace foo_nested_inline_ns_2::inline foo2 {
  void f0_nested_inline_ns_2();
  void f1_nested_inline_ns_2();
}
void f0_nested_inline_ns_2() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_nested_inline_ns_2' shadows 'foo_nested_inline_ns_2::foo2::f0_nested_inline_ns_2' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_nested_inline_ns_2::foo2::f0_nested_inline_ns_2() {}
void f1_nested_inline_ns_2() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_nested_inline_ns_2' shadows 'foo_nested_inline_ns_2::foo2::f1_nested_inline_ns_2' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////
