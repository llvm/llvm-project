// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify %s

typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Bool v8b __attribute__((ext_vector_type(8)));

@interface Obj @end

void objc(v8b mask, __strong Obj * ptr, v8i v) {
  (void)__builtin_masked_load(mask, ptr); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_store(mask, v, ptr); // expected-error {{3rd argument must be a scalar pointer}}
  (void)__builtin_masked_expand_load(mask, ptr); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_compress_store(mask, v, ptr); // expected-error {{3rd argument must be a scalar pointer}}
  (void)__builtin_masked_gather(mask, v, ptr); // expected-error {{3rd argument must be a scalar pointer}}
  (void)__builtin_masked_scatter(mask, v, v, ptr); // expected-error {{4th argument must be a scalar pointer}}
}
