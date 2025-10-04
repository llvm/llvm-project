// RUN: %check_clang_tidy %s modernize-use-using %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-using.IgnoreMacros: false}}"

#define CODE typedef int INT

CODE;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define CODE typedef int INT
// CHECK-FIXES: CODE;

struct Foo;

#define TYPEDEF typedef
TYPEDEF Foo Bak;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define TYPEDEF typedef
// CHECK-FIXES: TYPEDEF Foo Bak;
