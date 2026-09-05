// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// https://github.com/llvm/llvm-project/issues/173464
// A linkage query before redeclaration merging must not cache a result that
// depends on a previous declaration.
typedef float v4f __attribute__((vector_size(16)));
extern const v4f kAlias;
const v4f kAlias __attribute__((alias("kOne")));

// https://github.com/llvm/llvm-project/issues/112737
#pragma redefine_extname foo_cpp bar_cpp
static int foo_cpp(); // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_cpp'}}
extern int foo_cpp() { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_cpp'}}
