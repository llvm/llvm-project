// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

typedef char v8 __attribute__((ext_vector_type(8)));
typedef float v4 __attribute__((ext_vector_type(4)));

static long fake_oracle(void) {
  return 1;
}

[[clang::oracle_function]] static long oracle(void) {
  return 1;
}

[[clang::oracle_function]] static long oracle_with_params(int a) {
  return a;
}

void test_speculative_load(v8 *to_load, v4 *to_load2) {
  int a = 5;
  (void)__builtin_speculative_load(to_load); // expected-error {{too few arguments to function call}}
  (void)__builtin_speculative_load(5, 1, oracle); // expected-error {{indirection requires pointer operand ('int' invalid)}}
  (void)__builtin_speculative_load(to_load, a, oracle); // expected-error {{argument to '__builtin_speculative_load' must be a constant integer}}
  (void)__builtin_speculative_load(to_load, 0, 5); // expected-error {{the 3rd argument to '__builtin_speculative_load' must be a function}}
  (void)__builtin_speculative_load(to_load, 0, fake_oracle); // expected-error {{the oracle function passed to '__builtin_speculative_load' must have the oracle_function attribute}}
  (void)__builtin_speculative_load(to_load, 0, oracle_with_params); // expected-error {{expected to have 1 arguments after the oracle function to match its parameter count, but there are 0}}
  (void)__builtin_speculative_load(to_load, 0, oracle_with_params, "a"); // expected-error {{incompatible pointer to integer conversion passing 'char *' to parameter of type 'int'}} expected-note@14 {{passing argument to parameter 'a' here}}
  (void)__builtin_speculative_load(to_load2, 0, oracle_with_params, a); // expected-error {{the 1st argument to '__builtin_speculative_load' must be a pointer to a vector of integers}}
  (void)__builtin_speculative_load(to_load, 0, oracle_with_params, a);
}
