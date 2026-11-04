// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

using v8i = int [[clang::ext_vector_type(8)]];
using v8b = bool [[clang::ext_vector_type(8)]];

template <class T, class V>
static void load(v8b mask, V value, const T *ptr) {
  (void)__builtin_masked_load(mask, ptr, value); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_expand_load(mask, ptr, value); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_gather(mask, value, ptr); // expected-error {{3rd argument must be a scalar pointer}}
}

template <class T, class V>
static void store(v8b mask, V value, T *ptr) {
  (void)__builtin_masked_store(mask, value, ptr); // expected-error {{3rd argument must be a scalar pointer}}
  (void)__builtin_masked_compress_store(mask, value, ptr); // expected-error {{3rd argument must be a scalar pointer}}
  (void)__builtin_masked_scatter(mask, value, value, ptr); // expected-error {{4th argument must be a scalar pointer}}
}

void test_masked(v8b mask, v8i v, int *ptr) {
  load(mask, v, ptr);
  store(mask, v, ptr);
  load(mask, v, &v); // expected-note {{in instantiation of function template specialization 'load<int __attribute__((ext_vector_type(8))), int __attribute__((ext_vector_type(8)))>' requested here}}
  store(mask, v, &v); // expected-note {{in instantiation of function template specialization 'store<int __attribute__((ext_vector_type(8))), int __attribute__((ext_vector_type(8)))>' requested here}}
}
