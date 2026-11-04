// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

struct A { };

class B { using u = A; const static int p = 1; public: u *foo(); };

B::u* foo() { return nullptr; } 
// CHECK: error: 'u' is a private member of 'B'

B::u * B::foo() { return nullptr; }
// CHECK-NOT: error: 'u' is a private member of 'B'

class C { struct S { S(); ~S(); int f(); bool operator!() const; }; };

C::S::S() {}
// CHECK-NOT: error: 'S' is a private member of 'C'

C::S::~S() {}
// CHECK-NOT: error: 'S' is a private member of 'C'

int C::S::f() { return 0; }
// CHECK-NOT: error: 'S' is a private member of 'C'

bool C::S::operator!() const { return true; }
// CHECK-NOT: error: 'S' is a private member of 'C'

int p = B::p;
// CHECK: error: 'p' is a private member of 'B'
