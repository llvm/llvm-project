// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only \
// RUN:   -verify %s

using v4i = int __attribute__((ext_vector_type(4)));
using v4b = bool __attribute__((ext_vector_type(4)));

void ok_int_ref() {
  v4i v = {1, 2, 3, 4};
  int &r0 = v[0]; // ok
  int &r3 = v[3]; // ok
  (void)r0;
  (void)r3;
}

void bad_bool_ref(v4b vb) {
  bool &br = vb[1]; // expected-error {{non-const reference cannot bind to vector element}}
  (void)br;
}

void ok_const_bool_ref(v4b vb) {
  const bool &cr = vb[2]; // ok: binds to a temporary
  (void)cr;
}
