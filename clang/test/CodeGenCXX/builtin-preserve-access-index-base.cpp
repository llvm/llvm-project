// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify=ok %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DSECOND_BASE %s -o /dev/null
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -verify -DVIRTUAL_BASE %s -o /dev/null

// ok-no-diagnostics

// A CO-RE relocation is relative to the record that has the field, so an
// access through a base class must not need the offset of that base.

struct A { int a; };
struct B { int b; };
struct D : A, B { int d; };
struct V : virtual A { int v; };

unsigned long first_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->a);
}

unsigned long own_field(D *p, V *q) {
  return (unsigned long)__builtin_preserve_access_index(&p->d) +
         __builtin_preserve_field_info(q->v, 0);
}

int outside(D *p, V *q) { return p->b + q->a; }

#ifdef SECOND_BASE
unsigned long second_base(D *p) {
  return (unsigned long)__builtin_preserve_access_index(&p->b); // expected-error {{CO-RE access through a virtual base or a base at a non-zero offset not supported}}
}
#endif

#ifdef VIRTUAL_BASE
unsigned virtual_base(V *p) {
  return __builtin_preserve_field_info(p->a, 0); // expected-error {{CO-RE access through a virtual base or a base at a non-zero offset not supported}}
}
#endif
