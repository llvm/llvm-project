// RUN: not %clang_cc1 -triple %ms_abi_triple -ast-print %s -std=gnu++11 \
// RUN:     | FileCheck %s

// The test compiles a file with a syntax error which used to cause a crash with
// -ast-print. Compilation fails due to the syntax error, but compiler should
// not crash and print out whatever it manager to parse.

// CHECK:      struct {
// CHECK-NEXT: } dont_crash_on_syntax_error;
// CHECK-NEXT: decltype(nullptr) p;
struct {
} dont_crash_on_syntax_error /* missing ; */ decltype(nullptr) p;
