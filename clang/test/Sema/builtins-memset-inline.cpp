// RUN: %clang_cc1 -fsyntax-only -verify %s

#define NULL ((char *)0)

#if __has_builtin(__builtin_memset_inline)
#warning defined as expected
// expected-warning@-1 {{defined as expected}}
#endif

void test_memset_inline_invalid_arg_types() {
  __builtin_memset_inline(1, 2, 3); // expected-error {{cannot initialize a parameter of type 'void *' with an rvalue of type 'int'}}
}

void test_memset_inline_null_dst(void *ptr) {
  __builtin_memset_inline(NULL, 1, 4); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

void test_memset_inline_null_buffer_is_ok_if_size_is_zero(void *ptr, char value) {
  __builtin_memset_inline(NULL, value, /*size */ 0);
}

void test_memset_inline_non_constant_size(void *dst, char value, unsigned size) {
  __builtin_memset_inline(dst, value, size); // expected-error {{argument to '__builtin_memset_inline' must be a constant integer}}
}

template <unsigned size>
void test_memset_inline_template(void *dst, char value) {
  // we do not try to evaluate size in non intantiated templates.
  __builtin_memset_inline(dst, value, size);
}

void test_memset_inline_implicit_conversion(void *ptr, char value) {
  char a[5];
  __builtin_memset_inline(a, value, 5);
}

void test_memset_inline_num_args(void *dst, char value) {
  __builtin_memset_inline();                    // expected-error {{too few arguments to function call}}
  __builtin_memset_inline(dst, value, 4, NULL); // expected-error {{too many arguments to function call}}
}
