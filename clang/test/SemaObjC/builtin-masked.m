// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s

// expected-no-diagnostics
typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Bool v8b __attribute__((ext_vector_type(8)));

__attribute__((objc_root_class))
@interface Obj
@property int *ptr;
@end

void good(v8b mask, Obj *ptr, v8i v) {
  (void)__builtin_masked_load(mask, ptr.ptr);
  (void)__builtin_masked_store(mask, v, ptr.ptr);
  (void)__builtin_masked_expand_load(mask, ptr.ptr);
  (void)__builtin_masked_compress_store(mask, v, ptr.ptr);
  (void)__builtin_masked_gather(mask, v, ptr.ptr);
  (void)__builtin_masked_scatter(mask, v, v, ptr.ptr);
}
