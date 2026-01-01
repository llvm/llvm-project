// RUN: %check_clang_tidy -std=c++20 %s misc-use-internal-linkage %t -- -- -I%S/Inputs/use-internal-linkage

consteval void gh122096() {}

constexpr void cxf() {}
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: function 'cxf' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static constexpr void cxf() {}
