// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -faltivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -fsyntax-only   \
// RUN: -Wall -Werror -verify %s

// RUN: %clang_cc1 -faltivec -target-feature +power9-vector  \
// RUN: -triple powerpc64le-unknown-unknown -fsyntax-only    \
// RUN: -Wall -Werror -verify %s

#include <altivec.h>

extern vector signed int vsi;
extern vector unsigned char vuc;

void testInsertWord1(void) {
  int index = 5;
  vector unsigned char v1 = vec_insert4b(vsi, vuc, index); // expected-error {{argument to '__builtin_vsx_insertword' must be a constant integer}}
  vector unsigned long long v2 = vec_extract4b(vuc, index);   // expected-error {{argument to '__builtin_vsx_extractuword' must be a constant integer}}
}
