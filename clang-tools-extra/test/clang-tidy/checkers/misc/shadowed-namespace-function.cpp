// RUN: %check_clang_tidy -std=c++98-or-later %s misc-shadowed-namespace-function %t

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

namespace foo_op_general {
struct S1_general {};
}

void operator-(foo_op_general::S1_general, foo_op_general::S1_general);

namespace foo_op_general {
  void operator+(S1_general, S1_general);
  void operator-(S1_general, S1_general);
}
void operator+(foo_op_general::S1_general, foo_op_general::S1_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator+' shadows 'foo_op_general::operator+' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_op_general::operator+(foo_op_general::S1_general, foo_op_general::S1_general) {}
void operator-(foo_op_general::S1_general, foo_op_general::S1_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator-' shadows 'foo_op_general::operator-' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_op_lshift_general {
struct S1_lshift_general {};
struct S2_lshift_general {};
}

void operator<<(foo_op_lshift_general::S2_lshift_general, foo_op_lshift_general::S2_lshift_general);

namespace foo_op_lshift_general {
  void operator<<(S1_lshift_general, S1_lshift_general);
  void operator<<(S2_lshift_general, S2_lshift_general);
}
void operator<<(foo_op_lshift_general::S1_lshift_general, foo_op_lshift_general::S1_lshift_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator<<' shadows 'foo_op_lshift_general::operator<<' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_op_lshift_general::operator<<(foo_op_lshift_general::S1_lshift_general, foo_op_lshift_general::S1_lshift_general) {}
void operator<<(foo_op_lshift_general::S2_lshift_general, foo_op_lshift_general::S2_lshift_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator<<' shadows 'foo_op_lshift_general::operator<<' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_op_eq_general {
struct S1_eq_general {};
struct S2_eq_general {};
}

void operator==(foo_op_eq_general::S2_eq_general, foo_op_eq_general::S2_eq_general);

namespace foo_op_eq_general {
  void operator==(S1_eq_general, S1_eq_general);
  void operator==(S2_eq_general, S2_eq_general);
}
void operator==(foo_op_eq_general::S1_eq_general, foo_op_eq_general::S1_eq_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator==' shadows 'foo_op_eq_general::operator==' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_op_eq_general::operator==(foo_op_eq_general::S1_eq_general, foo_op_eq_general::S1_eq_general) {}
void operator==(foo_op_eq_general::S2_eq_general, foo_op_eq_general::S2_eq_general) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator==' shadows 'foo_op_eq_general::operator==' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

void f1_ambiguous();
namespace foo_ambiguous {
  void f0_ambiguous();
  void f1_ambiguous();
}
namespace foo_ambiguous2 {
  void f0_ambiguous();
  void f1_ambiguous();
}
void f0_ambiguous() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_ambiguous' shadows at least 'foo_ambiguous::f0_ambiguous' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
void f1_ambiguous() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_ambiguous' shadows at least 'foo_ambiguous::f1_ambiguous' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

void f1_ambiguous_nested();
namespace foo_ambiguous_nested::foo2::foo3 {
  void f0_ambiguous_nested();
  void f1_ambiguous_nested();
}
namespace foo_ambiguous_nested::foo2::foo4 {
  void f0_ambiguous_nested();
  void f1_ambiguous_nested();
}
void f0_ambiguous_nested() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_ambiguous_nested' shadows at least 'foo_ambiguous_nested::foo2::foo3::f0_ambiguous_nested' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
void f1_ambiguous_nested() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_ambiguous_nested' shadows at least 'foo_ambiguous_nested::foo2::foo3::f1_ambiguous_nested' [misc-shadowed-namespace-function]
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

template<typename T>
void f0_template() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_template' shadows 'foo::f0_template' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo::f0_template() {}
template<typename T>
void f1_template() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_template' shadows 'foo::f1_template' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

static void f1_static();
namespace foo_static {
  void f0_static();
  void f1_static();
}
static void f0_static() {}
static void f1_static() {}

//////////////////////////////////////////////////////////////////////////////////////////

void f1_static2();
namespace foo_static2 {
  void f0_static2();
  void f1_static2();
}
void f0_static2() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_static2' shadows 'foo_static2::f0_static2' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_static2::f0_static2() {}
void f1_static2() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_static2' shadows 'foo_static2::f1_static2' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

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
short f0_good4() { return 0; }
int f1_good4() { return 0; }

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_friend { struct A; }
void f1_friend(foo_friend::A);

namespace foo_friend {
  struct A{
    friend void f0_friend(A);
    friend void f1_friend(A);
  };
}

void f0_friend(foo_friend::A) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_friend' shadows 'foo_friend::f0_friend' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
void f1_friend(foo_friend::A) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_friend' shadows 'foo_friend::f1_friend' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

namespace foo_friend_op { struct A; }
void operator+(foo_friend_op::A, foo_friend_op::A);

namespace foo_friend_op {
  struct A{
    friend void operator+(A, A);
    friend void operator-(A, A);
  };
}

void operator+(foo_friend_op::A, foo_friend_op::A) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator+' shadows 'foo_friend_op::operator+' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
void operator-(foo_friend_op::A, foo_friend_op::A) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'operator-' shadows 'foo_friend_op::operator-' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

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

void f1_inline_ns();
inline namespace foo_inline_ns {
  void f0_inline_ns();
  void f1_inline_ns();
}
void f0_inline_ns() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_inline_ns' shadows 'foo_inline_ns::f0_inline_ns' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_inline_ns::f0_inline_ns() {}
void f1_inline_ns() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_inline_ns' shadows 'foo_inline_ns::f1_inline_ns' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

// Test case: The implementation only checks the first shadowed function for
// having a definition. If the first one has a definition, it returns early
// even if other shadowed functions without definitions should still warn.
void f1_definition_check();
namespace foo_def1 {
  void f0_definition_check();  // Has definition - found first
  void f1_definition_check();  // Has definition - found first
}
namespace foo_def2 {
  void f0_definition_check();     // No definition - should still warn
  void f1_definition_check();     // No definition - should still warn
}
void foo_def1::f0_definition_check() {}
void foo_def1::f1_definition_check() {}
void f0_definition_check() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f0_definition_check' shadows 'foo_def2::f0_definition_check' [misc-shadowed-namespace-function]
// CHECK-FIXES: void foo_def2::f0_definition_check() {}
void f1_definition_check() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: free function 'f1_definition_check' shadows 'foo_def2::f1_definition_check' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

//////////////////////////////////////////////////////////////////////////////////////////

namespace llvm {
class buffer_unique_ostream  {
  void anchor();
};
}

void anchor() {}

//////////////////////////////////////////////////////////////////////////////////////////
