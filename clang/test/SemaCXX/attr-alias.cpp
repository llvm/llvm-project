// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++17 -emit-llvm-only -verify %s
// expected-no-diagnostics

// Note: this mimics how interceptor functions are defined in the compiler-rt
// libraries. Despite a declaration in the system header having an exception
// specification, redeclaring it in the user code does not produce the "missing
// exception specification" error. Consequently, there should be no warnings
// about type mismatches for the alias and its aliasee.
# 1 "attr-alias.h" 1 3
extern "C" void test1() noexcept(true);
# 12 "attr-alias.cpp" 2
extern "C" void test1() __attribute__((alias("test1_aliasee")));
extern "C" void test1_aliasee() { }
