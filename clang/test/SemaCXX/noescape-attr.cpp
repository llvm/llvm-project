// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
void test1(T __attribute__((noescape)) arr, int size);

void test2(int __attribute__((noescape)) a, int b); // expected-warning {{'noescape' attribute only applies to a pointer, reference, class, struct, or union (0 is invalid)}}

struct S { int *p; };
void test3(S __attribute__((noescape)) s);

#if !__has_feature(attribute_noescape_nonpointer)
  #error "attribute_noescape_nonpointer should be supported"
#endif
