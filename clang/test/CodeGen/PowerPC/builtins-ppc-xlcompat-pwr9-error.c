// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s

extern unsigned int ui;
extern unsigned long long ull;
extern long long ll;
extern float f;
extern double d;

void test_builtin_ppc_cmprb() {
  int res = __builtin_ppc_cmprb(3, ui, ui); // expected-error {{argument value 3 is outside the valid range [0, 1]}}
}

#ifdef __PPC64__

void test_builtin_ppc_addex() {
  long long res = __builtin_ppc_addex(ll, ll, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  unsigned long long res2 = __builtin_ppc_addex(ull, ull, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
}

#endif

int test_builtin_ppc_test_data_class_d() {
  return __builtin_ppc_test_data_class(d, -1); // expected-error {{argument value -1 is outside the valid range [0, 127]}}
}

int test_builtin_ppc_test_data_class_f() {
  return __builtin_ppc_test_data_class(f, -1); // expected-error {{argument value -1 is outside the valid range [0, 127]}}
}

int test_test_data_class_d() {
  return __test_data_class(d, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
}

int test_test_data_class_f() {
  return __test_data_class(f, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
}

int test_test_data_class_type() {
  return __test_data_class(ui, 0); // expected-error {{expected a 'float' or 'double' for the first argument}}
}
