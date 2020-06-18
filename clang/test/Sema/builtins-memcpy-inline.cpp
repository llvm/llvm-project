// RUN: %clang_cc1 -fsyntax-only -verify %s

#define NULL ((char *)0)

#if __has_builtin(__builtin_memcpy_inline)
#warning defined as expected
// expected-warning@-1 {{defined as expected}}
#endif

void test_memcpy_inline_null_src(void *ptr) {
  __builtin_memcpy_inline(ptr, NULL, 4); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

void test_memcpy_inline_null_dst(void *ptr) {
  __builtin_memcpy_inline(NULL, ptr, 4); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

void test_memcpy_inline_null_buffers() {
  __builtin_memcpy_inline(NULL, NULL, 4);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}
  // expected-warning@-2 {{null passed to a callee that requires a non-null argument}}
}

void test_memcpy_inline_null_buffer_is_ok_if_size_is_zero(void *ptr) {
  __builtin_memcpy_inline(ptr, NULL, /*size */ 0);
  __builtin_memcpy_inline(NULL, ptr, /*size */ 0);
  __builtin_memcpy_inline(NULL, NULL, /*size */ 0);
}

void test_memcpy_inline_non_constant_size(void *dst, const void *src, unsigned size) {
  __builtin_memcpy_inline(dst, src, size); // expected-error {{argument to '__builtin_memcpy_inline' must be a constant integer}}
}

template <unsigned size>
void test_memcpy_inline_template(void *dst, const void *src) {
  // we do not try to evaluate size in non intantiated templates.
  __builtin_memcpy_inline(dst, src, size);
}
