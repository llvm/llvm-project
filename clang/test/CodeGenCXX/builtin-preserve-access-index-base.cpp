// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify=ok %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DFIRST_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DSECOND_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DVIRTUAL_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -debug-info-kind=limited -verify=ok -DFIRST_BASE -DSECOND_BASE -DVIRTUAL_BASE %s -o - | FileCheck %s

// ok-no-diagnostics

// A BPF CO-RE relocation is relative to the record that has the field, so an
// access through a base class would not relocate the offset of the base, which
// may differ in the target, even when it is 0 here. Other targets do not
// relocate and accept these accesses.

struct A { int a; };
struct B { int b; };
struct D : A, B { int d; };
struct V : virtual A { int v; };

unsigned long own_field(D *p, V *q) {
  return (unsigned long)__builtin_preserve_access_index(&p->d) +
         (unsigned long)__builtin_preserve_access_index(&q->v);
}

int outside(D *p, V *q) { return p->a + p->b + q->a; }

#ifdef FIRST_BASE
// CHECK-LABEL: @_Z10first_baseP1D(
unsigned long first_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->a); // expected-error {{CO-RE access through a base class not supported}}
}
#endif

#ifdef SECOND_BASE
// CHECK-LABEL: @_Z11second_baseP1D(
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
unsigned long second_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->b); // expected-error {{CO-RE access through a base class not supported}}
}
#endif

#ifdef VIRTUAL_BASE
// CHECK-LABEL: @_Z12virtual_baseP1V(
unsigned long virtual_base(V *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->a); // expected-error {{CO-RE access through a base class not supported}}
}
#endif
