// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-gnu -fpack-struct=1 -DPACK_STRUCT -verify=pack-struct %s
// RUN: %clang_cc1 -fsyntax-only -triple i386-apple-darwin -DMAC68K -verify=mac68k %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-gnu -target-feature +avx512f -DPACK16 -verify=itanium %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-windows-msvc -target-feature +avx512f -DPACK16 -verify=ms %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-gnu -DTYPEDEF -verify=typedef-itanium %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-windows-msvc -DTYPEDEF -verify=typedef-ms %s

extern void f1(int *);

#ifdef PACK_STRUCT
struct S {
  char c;
  int x;
};

void g0(struct S *p) {
  f1(&p->x); // pack-struct-warning {{packed member 'x' of class or structure 'S'}}
}

// #pragma pack overrides -fpack-struct.
#pragma pack(push, 4)
struct Pack4 {
  char c;
  int x;
};
#pragma pack(pop)

void g1(struct Pack4 *p) {
  f1(&p->x); // no-warning
}
#endif

#ifdef MAC68K
#pragma options align=mac68k
struct S {
  char c;
  int x;
};
#pragma options align=reset

void g0(struct S *p) {
  f1(&p->x); // mac68k-warning {{packed member 'x' of class or structure 'S'}}
}
#endif

#ifdef PACK16
typedef float V512 __attribute__((vector_size(64)));

// Only the Microsoft layout ignores a pack wider than a pointer.
#pragma pack(push, 16)
struct Pack16 {
  char c;
  V512 x;
};
#pragma pack(pop)

struct __attribute__((packed)) Outer {
  char c;
  struct Pack16 inner;
};

void g0(struct Outer *p) {
  V512 *q = &p->inner.x; // itanium-warning {{packed member 'x' of class or structure 'Pack16'}}
  // ms-warning@-1 {{packed member 'inner' of class or structure 'Outer'}}
}
#endif

#ifdef TYPEDEF
// typedef-ms-no-diagnostics
extern void f2(char *);

struct Natural {
  char c0;
  int x;
};
typedef struct Natural __attribute__((aligned(1))) UnalignedNatural;

// A typedef lowers alignment without any packing, but not in the Microsoft
// layout.
struct HoldsUnaligned {
  char c0;
  UnalignedNatural inner;
};

struct HoldsHoldsUnaligned {
  char c0;
  struct HoldsUnaligned holder;
};

typedef int __attribute__((aligned(1))) UnalignedInt;

struct HoldsUnalignedInt {
  char c0;
  UnalignedInt x;
};

void g0(struct HoldsUnaligned *p, struct HoldsHoldsUnaligned *q,
        struct HoldsUnalignedInt *r) {
  f1(&p->inner.x); // typedef-itanium-warning {{packed member 'inner' of class or structure 'HoldsUnaligned'}}
  f1(&q->holder.inner.x); // typedef-itanium-warning {{packed member 'inner' of class or structure 'HoldsUnaligned'}}
  f2(&p->inner.c0); // no-warning
  UnalignedInt *ui = &r->x; // no-warning
}
#endif
