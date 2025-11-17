// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

using my_int = int;
using my_short = short;
using my_short2 = short;

my_int f1(my_short2);
namespace foo {
  int f0(short);
  int f1(short);
}
my_int f0(my_short) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: free function 'f0' shadows 'foo::f0' [misc-shadowed-namespace-function]
// CHECK-FIXES: my_int foo::f0(my_short) {}
my_int f1(my_short) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: free function 'f1' shadows 'foo::f1' [misc-shadowed-namespace-function]
// CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
