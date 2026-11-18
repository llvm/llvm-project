// Testfile for (https://github.com/llvm/llvm-project/issues/216669)

// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +power9-vector -triple powerpc64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +power9-vector -triple powerpc64le-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +power9-vector -triple powerpc-unknown-unknown -fsyntax-only -verify %s

vector unsigned char test_bcdsetsign_too_many(vector unsigned char a) {
  return __builtin_ppc_bcdsetsign(a, 1, 2); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

vector unsigned char test_national2packed_too_many(vector unsigned char a) {
  return __builtin_ppc_national2packed(a, 1, 0); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

vector unsigned char test_packed2zoned_too_many(vector unsigned char a) {
  return __builtin_ppc_packed2zoned(a, 1, 0); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

vector unsigned char test_zoned2packed_too_many(vector unsigned char a) {
  return __builtin_ppc_zoned2packed(a, 1, 0); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

vector unsigned char test_bcdshift_too_many(vector unsigned char a) {
  return __builtin_ppc_bcdshift(a, 1, 0, 3); // expected-error {{too many arguments to function call, expected 3, have 4}}
}

vector unsigned char test_bcdshiftround_too_many(vector unsigned char a) {
  return __builtin_ppc_bcdshiftround(a, 1, 0, 3); // expected-error {{too many arguments to function call, expected 3, have 4}}
}

vector unsigned char test_bcdtruncate_too_many(vector unsigned char a) {
  return __builtin_ppc_bcdtruncate(a, 1, 0, 3); // expected-error {{too many arguments to function call, expected 3, have 4}}
}
