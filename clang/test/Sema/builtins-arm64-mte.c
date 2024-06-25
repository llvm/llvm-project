// RUN: %clang_cc1 -triple arm64-arm-eabi %s -target-feature +mte -fsyntax-only -verify
// RUN: %clang_cc1 -triple arm64-arm-eabi %s -target-feature +mte -x c++ -fsyntax-only -verify
// RUN: %clang_cc1 -triple arm64-arm-eabi %s -DNO_MTE -x c++ -emit-llvm-only -verify
#include <stddef.h>
#include <arm_acle.h>

#ifndef NO_MTE
int  *create_tag1(int a, unsigned b) {
  // expected-error@+1 {{first argument of MTE builtin function must be a pointer ('int' invalid)}}
  return __arm_mte_create_random_tag(a,b);
}

int  *create_tag2(int *a, unsigned *b) {
  // expected-error@+1 {{second argument of MTE builtin function must be an integer type ('unsigned int *' invalid)}}
  return __arm_mte_create_random_tag(a,b);
}

int  *create_tag3(const int *a, unsigned b) {
#ifdef __cplusplus
  // expected-error@+1 {{cannot initialize return object of type 'int *' with an rvalue of type 'const int *'}}
  return __arm_mte_create_random_tag(a,b);
#else
  // expected-warning@+1 {{returning 'const int *' from a function with result type 'int *' discards qualifiers}}
  return __arm_mte_create_random_tag(a,b);
#endif
}

int  *create_tag4(volatile int *a, unsigned b) {
#ifdef __cplusplus
  // expected-error@+1 {{cannot initialize return object of type 'int *' with an rvalue of type 'volatile int *'}}
  return __arm_mte_create_random_tag(a,b);
#else
  // expected-warning@+1 {{returning 'volatile int *' from a function with result type 'int *' discards qualifiers}}
  return __arm_mte_create_random_tag(a,b);
#endif
}

int  *increment_tag1(int *a, unsigned b) {
  // expected-error@+1 {{argument to '__builtin_arm_addg' must be a constant integer}}
  return __arm_mte_increment_tag(a,b);
}

int  *increment_tag2(int *a) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  return __arm_mte_increment_tag(a,16);
}

int  *increment_tag3(int *a) {
  // expected-error@+1 {{argument value -1 is outside the valid range [0, 15]}}
  return __arm_mte_increment_tag(a,-1);
}

int  *increment_tag4(const int *a) {
#ifdef __cplusplus
  // expected-error@+1 {{cannot initialize return object of type 'int *' with an rvalue of type 'const int *'}}
  return __arm_mte_increment_tag(a,5);
#else
  // expected-warning@+1 {{returning 'const int *' from a function with result type 'int *' discards qualifiers}}
  return __arm_mte_increment_tag(a,5);
#endif
}

int *increment_tag5(const volatile int *a) {
#ifdef __cplusplus
  // expected-error@+1 {{cannot initialize return object of type 'int *' with an rvalue of type 'const volatile int *'}}
  return __arm_mte_increment_tag(a,5);
#else
  // expected-warning@+1 {{returning 'const volatile int *' from a function with result type 'int *' discards qualifiers}}
  return __arm_mte_increment_tag(a,5);
#endif
}

unsigned exclude_tag1(int *ptr, unsigned m) {
   // expected-error@+1 {{first argument of MTE builtin function must be a pointer ('int' invalid)}}
   return  __arm_mte_exclude_tag(*ptr, m);
}

unsigned exclude_tag2(int *ptr, int *m) {
   // expected-error@+1 {{second argument of MTE builtin function must be an integer type ('int *' invalid)}}
   return  __arm_mte_exclude_tag(ptr, m);
}

void get_tag1(void) {
   // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
   __arm_mte_get_tag();
}

int *get_tag2(int ptr) {
   // expected-error@+1 {{first argument of MTE builtin function must be a pointer ('int' invalid)}}
   return __arm_mte_get_tag(ptr);
}

int *get_tag3(const volatile int *ptr) {
#ifdef __cplusplus
  // expected-error@+1 {{cannot initialize return object of type 'int *' with an rvalue of type 'const volatile int *'}}
  return __arm_mte_get_tag(ptr);
#else
  // expected-warning@+1 {{returning 'const volatile int *' from a function with result type 'int *' discards qualifiers}}
  return __arm_mte_get_tag(ptr);
#endif
}

void set_tag1(void) {
   // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
   __arm_mte_set_tag();
}

void set_tag2(int ptr) {
   // expected-error@+1 {{first argument of MTE builtin function must be a pointer ('int' invalid)}}
   __arm_mte_set_tag(ptr);
}

ptrdiff_t subtract_pointers1(int a, int *b) {
  // expected-error@+1 {{first argument of MTE builtin function must be a null or a pointer ('int' invalid)}}
  return __arm_mte_ptrdiff(a, b);
}

ptrdiff_t subtract_pointers2(int *a, int b) {
  // expected-error@+1 {{second argument of MTE builtin function must be a null or a pointer ('int' invalid)}}
  return __arm_mte_ptrdiff(a, b);
}

ptrdiff_t subtract_pointers3(char *a, int *b) {
  // expected-error@+1 {{'char *' and 'int *' are not pointers to compatible types}}
  return __arm_mte_ptrdiff(a, b);
}

ptrdiff_t subtract_pointers4(int *a, char *b) {
  // expected-error@+1 {{'int *' and 'char *' are not pointers to compatible types}}
  return __arm_mte_ptrdiff(a, b);
}

#ifdef __cplusplus
ptrdiff_t subtract_pointers5() {
  // expected-error@+1 {{at least one argument of MTE builtin function must be a pointer ('std::nullptr_t', 'std::nullptr_t' invalid)}}
  return __arm_mte_ptrdiff(nullptr, nullptr);
}
#endif

#else
int *create_tag1(int *a, unsigned b) {
  // expected-error@+1 {{'__builtin_arm_irg' needs target feature mte}}
  return __arm_mte_create_random_tag(a,b);
}
#endif