// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -O2 | FileCheck %s

extern "C" int printf(const char*, ...);

struct A { int val; A(int v); ~A(); void f() const; };
A::A(int v) : val(v) { printf("A(%d), this = %p\n", val, this); }
A::~A() { printf("~A, this = %p, val = %d\n", this, val); }
void A::f() const { printf("f: this = %p, val = %d\n", this, val); }

const A a(1);
// CHECK: A(1), this = [[THIS:0x[0-9a-f]+]]
// The constructor must only be called once!
// CHECK-NOT: A(1)

a.f();
// CHECK-NEXT: f: this = [[THIS]], val = 1
a.f();
// CHECK-NEXT: f: this = [[THIS]], val = 1

%quit
// There must still be no other constructor!
// CHECK-NOT: A(1)

// At the end, we expect exactly one destructor call
// CHECK: ~A
// CHECK-SAME: this = [[THIS]], val = 1
// CHECK-NOT: ~A
