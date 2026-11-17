// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o /dev/null -verify=b256 -DB256
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o /dev/null -verify=b300 -DB300
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o /dev/null -verify=array -DARRAY
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o /dev/null -verify=union -DUNION

typedef _Bool b256 __attribute__((ext_vector_type(256)));
typedef _Bool b300 __attribute__((ext_vector_type(300)));

#ifdef B256
// An i256 is aligned to 16 bytes where the vector is aligned to 32.
struct S256 {
  b256 v; // b256-error {{ClangIR code gen Not Yet Implemented: getStorageType for bool vector whose storage integer is aligned differently from the vector}}
  int x;
};
void f256(struct S256 *s) { s->x = 1; }
#endif

#ifdef B300
// An i300 is aligned to 16 bytes where the vector is aligned to 64.
struct S300 {
  b300 v; // b300-error {{ClangIR code gen Not Yet Implemented: getStorageType for bool vector whose storage integer is aligned differently from the vector}}
  int x;
};
void f300(struct S300 *s) { s->x = 1; }
#endif

#ifdef ARRAY
struct T {
  b256 a[2]; // array-error {{ClangIR code gen Not Yet Implemented: getStorageType for bool vector whose storage integer is aligned differently from the vector}}
};
void g(struct T *t) { (void)t->a[0]; }
#endif

#ifdef UNION
union U {
  b256 v; // union-error {{ClangIR code gen Not Yet Implemented: getStorageType for bool vector whose storage integer is aligned differently from the vector}}
  int x;
};
void h(union U *u) { u->x = 1; }
#endif
