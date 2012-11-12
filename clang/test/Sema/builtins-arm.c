// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify -DTEST0 %s
// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify -DTEST1 %s
// RUN: %clang_cc1 -triple armv7 -target-abi apcs-gnu \
// RUN:   -fsyntax-only -verify -DTEST1 %s

#ifdef TEST0
void __clear_cache(char*, char*);
#endif

#ifdef TEST1
void __clear_cache(void*, void*);
#endif

#if defined(__ARM_PCS) || defined(__ARM_EABI__)
// va_list on ARM AAPCS is struct { void* __ap }.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";
  *(ptr.__ap) = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}
#else
// va_list on ARM apcs-gnu is void*.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";  // expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
  *(ptr.__ap) = '0';// expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
}

void test2() {
  __builtin_va_list ptr = "x";
  *ptr = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}

#endif
