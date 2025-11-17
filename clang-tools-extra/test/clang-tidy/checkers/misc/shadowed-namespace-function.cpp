// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

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

//////////////////////////////////////////////////////////////////////////////////////////

void f1_variadic(...);
namespace foo_variadic {
  void f0_variadic(...);
  void f1_variadic(...);
}

// FIXME: warning in these two cases??
// FIXME: fixit for f0??
void f0_variadic(...) {}
void f1_variadic(...) {}

//////////////////////////////////////////////////////////////////////////////////////////

using my_int = int;
using my_short = short;
using my_short2 = short;

my_int f1_using(my_short2);
namespace foo_using {
  int f0_using(short);
  int f1_using(short);
}
my_int f0_using(my_short) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: free function 'f0_using' shadows 'foo_using::f0_using' [misc-shadowed-namespace-function]
// CHECK-FIXES: my_int foo_using::f0_using(my_short) {}
my_int f1_using(my_short) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: free function 'f1_using' shadows 'foo_using::f1_using' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void f1_template();
namespace foo {
  template<typename T>
  void f0_template();
  template<typename T>
  void f1_template();
}

// FIXME: provide warning in these two cases
// FIXME: provide fixit for f0
template<typename T>
void f0_template() {}
template<typename T>
void f1_template() {}

//////////////////////////////////////////////////////////////////////////////////////////

static void f1_static();
namespace foo_static {
  void f0_static();
  void f1_static();
}
static void f0_static() {}
static void f1_static() {}

//////////////////////////////////////////////////////////////////////////////////////////

void f1_nested();
namespace foo_nested::foo2::foo3 {
  void f0_nested();
  void f1_nested();
}
void f0_nested() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_nested' shadows 'foo_nested::foo2::foo3::f0_nested' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_nested::foo2::foo3::f0_nested() {}
void f1_nested() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_nested' shadows 'foo_nested::foo2::foo3::f1_nested' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_main {
  int main();
}
int main() {}

//////////////////////////////////////////////////////////////////////////////////////////

#define VOID_F0 void f0_macro
#define VOID_F1_BRACES_BODY void f1_macro() {}

void f1_macro();
namespace foo_macro {
  void f0_macro();
  void f1_macro();
}
VOID_F0() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: free function 'f0_macro' shadows 'foo_macro::f0_macro' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
VOID_F1_BRACES_BODY
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: free function 'f1_macro' shadows 'foo_macro::f1_macro' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

void f1_good();
namespace foo_good {
  void f0_good() {}
  void f1_good() {}
}
void f0_good() {}
void f1_good() {}

//////////////////////////////////////////////////////////////////////////////////////////

void f1_good2();
namespace foo_good2 {
  void f0_good2();
  void f1_good2();
}
void foo_good2::f0_good2() {}
void foo_good2::f1_good2() {}
void f0_good2() {}
void f1_good2() {}

//////////////////////////////////////////////////////////////////////////////////////////

void f1_good3(int);
namespace foo_good3 {
  void f0_good3(short);
  void f1_good3(unsigned);
}
void f0_good3(char) {}
void f1_good3(int) {}

//////////////////////////////////////////////////////////////////////////////////////////

int f1_good4();
namespace foo_good4 {
  char f0_good4();
  unsigned f1_good4();
}
short f0_good4() { return {}; }
int f1_good4() { return {}; }

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_friend { struct A; }
void f1_friend(foo_friend::A);

namespace foo_friend {
  struct A{
    friend void f0_friend(A);
    friend void f1_friend(A);
  };
}

// FIXME: provide warning without fixit in these two cases
void f0_friend(foo_friend::A) {}
void f1_friend(foo_friend::A) {}

//////////////////////////////////////////////////////////////////////////////////////////

namespace { void f1_anon(); }
namespace foo_anon {
  void f0_anon();
  void f1_anon();
}
namespace {
void f0_anon() {}
void f1_anon() {}
}

//////////////////////////////////////////////////////////////////////////////////////////
