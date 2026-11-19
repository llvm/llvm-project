// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s --check-prefix=NEG

// A function template specialization that a failed input instantiates for the
// first time must be instantiated again by a later input using it. Otherwise
// the later input references a function the code generator never saw, and the
// JIT fails to resolve its symbol.

extern "C" int printf(const char *, ...);

template <typename T> T plusOne(T t) { return t + 1; }

int f() { return plusOne(1) }
// CHECK-DAG: error: expected ';' after return statement
int f() { return plusOne(1); }
auto r1 = printf("f() = %d\n", f());
// CHECK-DAG: f() = 2

int g() { return plusOne(2L) + undeclared_thing; }
// CHECK-DAG: error: use of undeclared identifier 'undeclared_thing'
long g() { return plusOne(2L); }
auto r2 = printf("g() = %ld\n", g());
// CHECK-DAG: g() = 3

template <typename T> inline T plusTwo(T t) { return t + 2; }
int h() { return plusTwo(1) + undeclared_thing; }
int h() { return plusTwo(1); }
auto r3 = printf("h() = %d\n", h());
// CHECK-DAG: h() = 3

// A member function of a class template specialization from an earlier input.
template <typename T> struct S { T get() { return T(4); } };
S<int> s;
int k() { return s.get() + undeclared_thing; }
int k() { return s.get(); }
auto r4 = printf("k() = %d\n", k());
// CHECK-DAG: k() = 4

// NEG-NOT: Symbols not found
// NEG-NOT: is not defined

%quit
