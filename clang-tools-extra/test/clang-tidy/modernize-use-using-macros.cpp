// RUN: %check_clang_tidy %s modernize-use-using %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-using.IgnoreMacros, value: 0}]}"

#define CODE typedef int INT

CODE;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define CODE typedef int INT
// CHECK-FIXES: CODE;

struct Foo;
#define Bar Baz
typedef Foo Bar;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define Bar Baz
// CHECK-FIXES: using Baz = Foo;

#define TYPEDEF typedef
TYPEDEF Foo Bak;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define TYPEDEF typedef
// CHECK-FIXES: TYPEDEF Foo Bak;
