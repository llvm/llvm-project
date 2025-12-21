// RUN: %check_clang_tidy -std=c++98-or-later %s misc-shadowed-namespace-function %t -- \
// RUN:   --config="{CheckOptions: {misc-shadowed-namespace-function.IgnoreTemplated: true}}"

namespace foo {
  template<typename T>
  void f_template();
}

template<typename T>
void f_template() {}
// When IgnoreTemplated is true, the warning should NOT appear

template void f_template<int>();

//////////////////////////////////////////////////////////////////////////////////////////

namespace bar {
  template<typename T>
  void g_template();
}

template<typename T>
void g_template() {}
// When IgnoreTemplated is true, the warning should NOT appear

const int _ = (g_template<char>(), 0);

//////////////////////////////////////////////////////////////////////////////////////////

// Test with another template
namespace bar2 {
  template<typename T>
  void j_template();
}

template<typename T>
void j_template() {}
// When IgnoreTemplated is true, the warning should NOT appear

//////////////////////////////////////////////////////////////////////////////////////////

void f1_general();
namespace foo_general {
  void f0_general();
  void f1_general();
}
void f0_general() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_general' shadows 'foo_general::f0_general' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_general::f0_general() {}
void f1_general() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_general' shadows 'foo_general::f1_general' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
