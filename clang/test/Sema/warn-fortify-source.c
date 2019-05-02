// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_PASS_OBJECT_SIZE
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_BUILTINS
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_PASS_OBJECT_SIZE
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_BUILTINS

typedef unsigned long size_t;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(USE_PASS_OBJECT_SIZE)
void *memcpy(void *dst, const void *src, size_t c);
static void *memcpy(void *dst __attribute__((pass_object_size(1))), const void *src, size_t c) __attribute__((overloadable)) __asm__("merp");
static void *memcpy(void *const dst __attribute__((pass_object_size(1))), const void *src, size_t c) __attribute__((overloadable)) {
  return 0;
}
#elif defined(USE_BUILTINS)
#define memcpy(x,y,z) __builtin_memcpy(x,y,z)
#else
void *memcpy(void *dst, const void *src, size_t c);
#endif

#ifdef __cplusplus
}
#endif

void call_memcpy() {
  char dst[10];
  char src[20];
  memcpy(dst, src, 20); // expected-warning {{memcpy' will always overflow; destination buffer has size 10, but size argument is 20}}

  if (sizeof(dst) == sizeof(src))
    memcpy(dst, src, 20); // no warning, unreachable
}

void call_memcpy_type() {
  struct pair {
    int first;
    int second;
  };
  struct pair p;
  char buf[20];
  memcpy(&p.first, buf, 20);
#ifdef USE_PASS_OBJECT_SIZE
  // Use the more strict checking mode on the pass_object_size attribute:
  // expected-warning@-3 {{memcpy' will always overflow; destination buffer has size 4, but size argument is 20}}
#else
  // Or just fallback to type 0:
  // expected-warning@-6 {{memcpy' will always overflow; destination buffer has size 8, but size argument is 20}}
#endif
}

void call_strncat() {
  char s1[10], s2[20];
  __builtin_strncat(s2, s1, 20);
  __builtin_strncat(s1, s2, 20); // expected-warning {{'strncat' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_strncpy() {
  char s1[10], s2[20];
  __builtin_strncpy(s2, s1, 20);
  __builtin_strncpy(s1, s2, 20); // expected-warning {{'strncpy' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_stpncpy() {
  char s1[10], s2[20];
  __builtin_stpncpy(s2, s1, 20);
  __builtin_stpncpy(s1, s2, 20); // expected-warning {{'stpncpy' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_memmove() {
  char s1[10], s2[20];
  __builtin_memmove(s2, s1, 20);
  __builtin_memmove(s1, s2, 20); // expected-warning {{'memmove' will always overflow; destination buffer has size 10, but size argument is 20}}
}

void call_memset() {
  char buf[10];
  __builtin_memset(buf, 0xff, 10);
  __builtin_memset(buf, 0xff, 11); // expected-warning {{'memset' will always overflow; destination buffer has size 10, but size argument is 11}}
}

void call_snprintf() {
  char buf[10];
  __builtin_snprintf(buf, 10, "merp");
  __builtin_snprintf(buf, 11, "merp"); // expected-warning {{'snprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
}

void call_vsnprintf() {
  char buf[10];
  __builtin_va_list list;
  __builtin_vsnprintf(buf, 10, "merp", list);
  __builtin_vsnprintf(buf, 11, "merp", list); // expected-warning {{'vsnprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
}

#ifdef __cplusplus
template <class> struct S {
  void mf() const {
    __builtin_memset(const_cast<char *>(mv), 0, 0);
  }

  char mv[10];
};

template <int A, int B>
void call_memcpy_dep() {
  char bufferA[A];
  char bufferB[B];
  memcpy(bufferA, bufferB, 10); // expected-warning{{'memcpy' will always overflow; destination buffer has size 9, but size argument is 10}}
}

void call_call_memcpy() {
  call_memcpy_dep<10, 9>();
  call_memcpy_dep<9, 10>(); // expected-note {{in instantiation of function template specialization 'call_memcpy_dep<9, 10>' requested here}}
}
#endif
