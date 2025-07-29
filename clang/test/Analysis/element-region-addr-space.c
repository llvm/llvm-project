// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN: -analyzer-checker=core,unix.Malloc,debug.ExprInspection -verify \
// RUN: -Wno-incompatible-pointer-types -Wno-unused-comparison %s

// expected-no-diagnostics
//
// By default, pointers are 64-bits.
#define ADDRESS_SPACE_32BITS __attribute__((address_space(3)))
ADDRESS_SPACE_32BITS void *b();
ADDRESS_SPACE_32BITS int *c();
typedef struct {
  ADDRESS_SPACE_32BITS int *e;
} f;
void g() {
  ADDRESS_SPACE_32BITS void *h = b();
  ADDRESS_SPACE_32BITS f *j = c();
  j->e == h;
}
