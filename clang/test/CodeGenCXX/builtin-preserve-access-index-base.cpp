// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify=ok %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DFIRST_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DSECOND_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DVIRTUAL_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DDERIVED %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DFIELD_INFO %s -o /dev/null
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -debug-info-kind=limited -verify=ok -DFIRST_BASE -DSECOND_BASE -DVIRTUAL_BASE -DDERIVED %s -o - | FileCheck %s

// ok-no-diagnostics

// A BPF CO-RE relocation is relative to the record that has the field, so an
// address computed through a conversion between a class and its base would not
// relocate the offset of the base, which may differ in the target, even when
// it is 0 here. Conversions that the address does not depend on are fine, and
// other targets do not relocate and accept all of these.

struct A { int a; };
struct B { int b; };
struct D : A, B { int x; };
struct V : virtual A { int v; };
void consume(A *);

unsigned long own_field(D *p, V *q) {
  return (unsigned long)__builtin_preserve_access_index(&p->x) +
         (unsigned long)__builtin_preserve_access_index(&q->v);
}

unsigned long side_effect(D *d) {
  return (unsigned long)__builtin_preserve_access_index((consume(d), &d->x));
}

int outside(D *p, V *q) { return p->a + p->b + q->a; }

#ifdef FIRST_BASE
// CHECK-LABEL: @_Z10first_baseP1D(
unsigned long first_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->a); // expected-error {{CO-RE access through a base class conversion not supported}}
}
#endif

#ifdef SECOND_BASE
// CHECK-LABEL: @_Z11second_baseP1D(
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
unsigned long second_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->b); // expected-error {{CO-RE access through a base class conversion not supported}}
}
#endif

#ifdef VIRTUAL_BASE
// CHECK-LABEL: @_Z12virtual_baseP1V(
unsigned long virtual_base(V *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->a); // expected-error {{CO-RE access through a base class conversion not supported}}
}
#endif

#ifdef DERIVED
// CHECK-LABEL: @_Z7derivedP1B(
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 -4
unsigned long derived(B *p) {
  return (unsigned long)__builtin_preserve_access_index(&static_cast<D *>(p)->x); // expected-error {{CO-RE access through a base class conversion not supported}}
}
#endif

#ifdef FIELD_INFO
unsigned field_info(D *p) {
  return __builtin_preserve_field_info(p->b, 0); // expected-error {{CO-RE access through a base class conversion not supported}}
}
#endif
