// RUN: %check_clang_tidy %s readability-named-parameter %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-named-parameter.IgnoredTypes, value: 'MyCustomTag'}]}"

struct MyCustomTag {};
struct OtherTag {};

// MyCustomTag is in the custom IgnoredTypes list, so it should not warn.
void f_custom(MyCustomTag) {}

// OtherTag is NOT in the custom list, so it should warn.
void f_other(OtherTag) {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: all parameters should be named in a function
// CHECK-FIXES: void f_other(OtherTag /*unused*/) {}

// int is not in the custom list either, so it should warn.
void f_int(int) {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: all parameters should be named in a function
// CHECK-FIXES: void f_int(int /*unused*/) {}

// Verify the default std types are NOT ignored when a custom list replaces them.
namespace std { struct in_place_t {}; }
void f_in_place(std::in_place_t) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: all parameters should be named in a function
// CHECK-FIXES: void f_in_place(std::in_place_t /*unused*/) {}
