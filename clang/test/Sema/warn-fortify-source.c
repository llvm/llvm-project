// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_BUILTINS
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify -DUSE_BUILTINS

typedef unsigned long size_t;

#ifdef __cplusplus
extern "C" {
#endif

extern int sprintf(char *str, const char *format, ...);

#if defined(USE_BUILTINS)
#define memcpy(x,y,z) __builtin_memcpy(x,y,z)
#else
void *memcpy(void *dst, const void *src, size_t c);
#endif

#ifdef __cplusplus
}
#endif

void call_memcpy(void) {
  char dst[10];
  char src[20];
  memcpy(dst, src, 20); // expected-warning {{memcpy' will always overflow; destination buffer has size 10, but size argument is 20}}

  if (sizeof(dst) == sizeof(src))
    memcpy(dst, src, 20); // no warning, unreachable
}

void call_memcpy_type(void) {
  struct pair {
    int first;
    int second;
  };
  struct pair p;
  char buf[20];
  memcpy(&p.first, buf, 20); // expected-warning {{memcpy' will always overflow; destination buffer has size 8, but size argument is 20}}
}

void call_strncat(void) {
  char s1[10], s2[20];
  __builtin_strncat(s2, s1, 20);
  __builtin_strncat(s1, s2, 20); // expected-warning {{'strncat' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_strncpy(void) {
  char s1[10], s2[20];
  __builtin_strncpy(s2, s1, 20);
  __builtin_strncpy(s1, s2, 20); // expected-warning {{'strncpy' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_stpncpy(void) {
  char s1[10], s2[20];
  __builtin_stpncpy(s2, s1, 20);
  __builtin_stpncpy(s1, s2, 20); // expected-warning {{'stpncpy' size argument is too large; destination buffer has size 10, but size argument is 20}}
}

void call_strcpy(void) {
  const char *const src = "abcd";
  char dst[4];
  __builtin_strcpy(dst, src); // expected-warning {{'strcpy' will always overflow; destination buffer has size 4, but the source string has length 5 (including NUL byte)}}
}

void call_strcpy_nowarn(void) {
  const char *const src = "abcd";
  char dst[5];
  // We should not get a warning here.
  __builtin_strcpy(dst, src);
}

void call_memmove(void) {
  char s1[10], s2[20];
  __builtin_memmove(s2, s1, 20);
  __builtin_memmove(s1, s2, 20); // expected-warning {{'memmove' will always overflow; destination buffer has size 10, but size argument is 20}}
}

void call_memset(void) {
  char buf[10];
  __builtin_memset(buf, 0xff, 10);
  __builtin_memset(buf, 0xff, 11); // expected-warning {{'memset' will always overflow; destination buffer has size 10, but size argument is 11}}
}

void call_snprintf(double d, int n) {
  char buf[10];
  __builtin_snprintf(buf, 10, "merp");
  __builtin_snprintf(buf, 11, "merp"); // expected-warning {{'snprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
  __builtin_snprintf(buf, 0, "merp");
  __builtin_snprintf(buf, 3, "merp"); // expected-warning {{'snprintf' will always be truncated; specified size is 3, but format string expands to at least 5}}
  __builtin_snprintf(buf, 4, "merp"); // expected-warning {{'snprintf' will always be truncated; specified size is 4, but format string expands to at least 5}}
  __builtin_snprintf(buf, 5, "merp");
  __builtin_snprintf(buf, 1, "%.1000g", d); // expected-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 5, "%.1000g", d);
  __builtin_snprintf(buf, 5, "%.1000G", d);
  __builtin_snprintf(buf, 10, " %#08x", n);
  __builtin_snprintf(buf, 2, "%#x", n);
  __builtin_snprintf(buf, 2, "%#X", n);
  __builtin_snprintf(buf, 2, "%#o", n);
  __builtin_snprintf(buf, 1, "%#x", n); // expected-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 1, "%#X", n); // expected-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 1, "%#o", n); // expected-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
}

void call_vsnprintf(void) {
  char buf[10];
  __builtin_va_list list;
  __builtin_vsnprintf(buf, 10, "merp", list);
  __builtin_vsnprintf(buf, 11, "merp", list); // expected-warning {{'vsnprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
  __builtin_vsnprintf(buf, 0, "merp", list);
  __builtin_vsnprintf(buf, 3, "merp", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 3, but format string expands to at least 5}}
  __builtin_vsnprintf(buf, 4, "merp", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 4, but format string expands to at least 5}}
  __builtin_vsnprintf(buf, 5, "merp", list);
  __builtin_vsnprintf(buf, 1, "%.1000g", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 5, "%.1000g", list);
  __builtin_vsnprintf(buf, 5, "%.1000G", list);
  __builtin_vsnprintf(buf, 10, " %#08x", list);
  __builtin_vsnprintf(buf, 2, "%#x", list);
  __builtin_vsnprintf(buf, 2, "%#X", list);
  __builtin_vsnprintf(buf, 2, "%#o", list);
  __builtin_vsnprintf(buf, 1, "%#x", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 1, "%#X", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 1, "%#o", list); // expected-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
}

void call_sprintf_chk(char *buf) {
  __builtin___sprintf_chk(buf, 1, 6, "hell\n");
  __builtin___sprintf_chk(buf, 1, 5, "hell\n");     // expected-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 6, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}}
  __builtin___sprintf_chk(buf, 1, 2, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}} \
                                                    // expected-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 5}}
  __builtin___sprintf_chk(buf, 1, 6, "hello");
  __builtin___sprintf_chk(buf, 1, 5, "hello"); // expected-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 2, "%c", '9');
  __builtin___sprintf_chk(buf, 1, 1, "%c", '9'); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%d", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%i", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%i", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%o", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%o", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%u", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%u", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%x", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%x", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%X", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%X", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%hhd", (char)9);
  __builtin___sprintf_chk(buf, 1, 1, "%hhd", (char)9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%hd", (short)9);
  __builtin___sprintf_chk(buf, 1, 1, "%hd", (short)9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%ld", 9l);
  __builtin___sprintf_chk(buf, 1, 1, "%ld", 9l); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%lld", 9ll);
  __builtin___sprintf_chk(buf, 1, 1, "%lld", 9ll); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%%");
  __builtin___sprintf_chk(buf, 1, 1, "%%"); // expected-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 4, "%#x", 9);
  __builtin___sprintf_chk(buf, 1, 3, "%#x", 9);
  __builtin___sprintf_chk(buf, 1, 4, "%p", (void *)9);
  __builtin___sprintf_chk(buf, 1, 3, "%p", (void *)9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 3, but format string expands to at least 4}}
  __builtin___sprintf_chk(buf, 1, 3, "%+d", 9);
  __builtin___sprintf_chk(buf, 1, 2, "%+d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 3}}
  __builtin___sprintf_chk(buf, 1, 3, "% i", 9);
  __builtin___sprintf_chk(buf, 1, 2, "% i", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 3}}
  __builtin___sprintf_chk(buf, 1, 6, "%5d", 9);
  __builtin___sprintf_chk(buf, 1, 5, "%5d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 9, "%f", 9.f);
  __builtin___sprintf_chk(buf, 1, 8, "%f", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 8, but format string expands to at least 9}}
  __builtin___sprintf_chk(buf, 1, 9, "%Lf", (long double)9.);
  __builtin___sprintf_chk(buf, 1, 8, "%Lf", (long double)9.); // expected-warning {{'sprintf' will always overflow; destination buffer has size 8, but format string expands to at least 9}}
  __builtin___sprintf_chk(buf, 1, 10, "%+f", 9.f);
  __builtin___sprintf_chk(buf, 1, 9, "%+f", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 9, but format string expands to at least 10}}
  __builtin___sprintf_chk(buf, 1, 12, "%e", 9.f);
  __builtin___sprintf_chk(buf, 1, 11, "%e", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 11, but format string expands to at least 12}}
}

void call_sprintf(void) {
  char buf[6];
  sprintf(buf, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}}
  sprintf(buf, "hello b\0y"); // expected-warning {{format string contains '\0' within the string body}} \
                              // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 8}}
  sprintf(buf, "hello");
  sprintf(buf, "hello!"); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%%");
  sprintf(buf, "12345%%"); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%c", '9');
  sprintf(buf, "12345%c", '9'); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%d", 9);
  sprintf(buf, "12345%d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%lld", 9ll);
  sprintf(buf, "12345%lld", 9ll); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "12%#x", 9);
  sprintf(buf, "123%#x", 9);
  sprintf(buf, "12%p", (void *)9);
  sprintf(buf, "123%p", (void *)9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "123%+d", 9);
  sprintf(buf, "1234%+d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "123% i", 9);
  sprintf(buf, "1234% i", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%5d", 9);
  sprintf(buf, "1%5d", 9); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%.3f", 9.f);
  sprintf(buf, "5%.3f", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%+.2f", 9.f);
  sprintf(buf, "%+.3f", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%.0e", 9.f);
  sprintf(buf, "5%.1e", 9.f); // expected-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 8}}
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
