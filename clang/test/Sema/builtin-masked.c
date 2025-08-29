// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Bool v8b __attribute__((ext_vector_type(8)));
typedef _Bool v2b __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

void test_masked_load(v8i *pf, v8b mask, v2b mask2, v2b thru) {
  (void)__builtin_masked_load(mask); // expected-error {{too few arguments to function call, expected 2, have 1}}
  (void)__builtin_masked_load(mask, pf, pf, pf); // expected-error {{too many arguments to function call, expected at most 3, have 4}}
  (void)__builtin_masked_load(mask2, pf); // expected-error {{all arguments to __builtin_masked_load must have the same number of elements}}
  (void)__builtin_masked_load(mask, mask); // expected-error {{2nd argument must be a pointer to vector}}
  (void)__builtin_masked_load(mask, (void *)0); // expected-error {{2nd argument must be a pointer to vector}}
  (void)__builtin_masked_load(mask2, pf, thru); // expected-error {{3rd argument must be a 'v8i' (vector of 8 'int' values)}}
  (void)__builtin_masked_load(mask2, pf); // expected-error {{all arguments to __builtin_masked_load must have the same number of elements}}
}

void test_masked_store(v8i *pf, v8f *pf2, v8b mask, v2b mask2) {
  __builtin_masked_store(mask); // expected-error {{too few arguments to function call, expected 3, have 1}}
  __builtin_masked_store(mask, 0, 0, 0); // expected-error {{too many arguments to function call, expected 3, have 4}}
  __builtin_masked_store(0, 0, pf); // expected-error {{1st argument must be a vector of boolean types (was 'int')}}
  __builtin_masked_store(mask, 0, pf); // expected-error {{2nd argument must be a vector}}
  __builtin_masked_store(mask, *pf, 0); // expected-error {{3rd argument must be a pointer to vector}}
  __builtin_masked_store(mask2, *pf, pf); // expected-error {{all arguments to __builtin_masked_store must have the same number of elements}}
  __builtin_masked_store(mask, *pf, pf2); // expected-error {{last two arguments to '__builtin_masked_store' must have the same type}}
}
