// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-va-opt %t

extern void foo(...);


// CHECK-MESSAGES: :[[@LINE+2]]:26: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-FIXES: #define M0(...) foo(1 __VA_OPT__(',') __VA_ARGS__)
#define M0(...) foo(1, ##__VA_ARGS__)

// CHECK-MESSAGES: :[[@LINE+2]]:27: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-FIXES: #define M1(...) foo(1 __VA_OPT__(',') __VA_ARGS__)
#define M1(...) foo(1, ## __VA_ARGS__)

// CHECK-MESSAGES: :[[@LINE+2]]:27: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-FIXES: #define M2(...) foo(1 __VA_OPT__(',') __VA_ARGS__)
#define M2(...) foo(1 ,## __VA_ARGS__)

// CHECK-MESSAGES: :[[@LINE+2]]:28: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-FIXES: #define M3(...) foo(1 __VA_OPT__(',') __VA_ARGS__)
#define M3(...) foo(1 , ## __VA_ARGS__)

// CHECK-MESSAGES: :[[@LINE+3]]:28: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-MESSAGES: :[[@LINE+2]]:44: warning: Use __VA_OPT__ instead of GNU extension to __VA_ARGS__ [modernize-use-va-opt]
// CHECK-FIXES: #define M4(...) foo(1 __VA_OPT__(',') __VA_ARGS__ __VA_OPT__(',') __VA_ARGS__)
#define M4(...) foo(1 , ## __VA_ARGS__, ## __VA_ARGS__)

// No message, this will never add a comma before __VA_ARGS__
#define P0(...) foo(1 ##__VA_ARGS__)

// No message, this will never add a comma before __VA_ARGS__
#define P1(...) foo(__VA_ARGS__)

// No message, this will never add a comma before __VA_ARGS__
#define P2(...) foo(##__VA_ARGS__)

// No message, this will never add a comma before __VA_ARGS__
#define P3(...) foo(, __VA_ARGS__)
