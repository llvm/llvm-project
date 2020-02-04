// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

struct A { void *p; void *q; void *r; };

struct B : A {};
static_assert(sizeof(B) == 24);

// CHECK: define dereferenceable(24) {{.*}} @_Z1fR1B({{.*}} dereferenceable(24)
B &f(B &b) { return b; }

struct C : virtual A {};
static_assert(sizeof(C) == 32);

// CHECK: define dereferenceable(8) {{.*}} @_Z1fR1C({{.*}} dereferenceable(8)
C &f(C &c) { return c; }
