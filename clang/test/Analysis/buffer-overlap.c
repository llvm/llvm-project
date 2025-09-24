// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap
//
// RUN: %clang_analyze_cc1 -verify %s -DUSE_BUILTINS \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap
//
// RUN: %clang_analyze_cc1 -verify %s -DVARIANT \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap
//
// RUN: %clang_analyze_cc1 -verify %s -DVARIANT -DUSE_BUILTINS \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap

// This provides us with four possible sprintf() definitions.

#ifdef USE_BUILTINS
#define BUILTIN(f) __builtin_##f
#else /* USE_BUILTINS */
#define BUILTIN(f) f
#endif /* USE_BUILTINS */

typedef typeof(sizeof(int)) size_t;

#ifdef VARIANT

#define __sprintf_chk BUILTIN(__sprintf_chk)
#define __snprintf_chk BUILTIN(__snprintf_chk)
int __sprintf_chk (char * __restrict str, int flag, size_t os,
        const char * __restrict fmt, ...);
int __snprintf_chk (char * __restrict str, size_t len, int flag, size_t os,
        const char * __restrict fmt, ...);

#define sprintf(str, ...) __sprintf_chk(str, 0, __builtin_object_size(str, 0), __VA_ARGS__)
#define snprintf(str, len, ...) __snprintf_chk(str, len, 0, __builtin_object_size(str, 0), __VA_ARGS__)

#else /* VARIANT */

#define sprintf BUILTIN(sprintf)
int sprintf(char *restrict buffer, const char *restrict format, ... );
int snprintf(char *restrict buffer, size_t bufsz,
             const char *restrict format, ... );
#endif /* VARIANT */

void test_sprintf1() {
  char a[4] = {0};
  sprintf(a, "%d/%s", 1, a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_sprintf2() {
  char a[4] = {0};
  sprintf(a, "%s", a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_sprintf3() {
  char a[4] = {0};
  sprintf(a, "%s/%s", a, a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_sprintf4() {
  char a[4] = {0};
  sprintf(a, "%d", 42); // no-warning
}

void test_sprintf5() {
  char a[4] = {0};
  char b[4] = {0};
  sprintf(a, "%s", b); // no-warning
}

void test_snprintf1() {
  char a[4] = {0};
  snprintf(a, sizeof(a), "%d/%s", 1, a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_snprintf2() {
  char a[4] = {0};
  snprintf(a+1, sizeof(a)-1, "%d/%s", 1, a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_snprintf3() {
  char a[4] = {0};
  snprintf(a, sizeof(a), "%s", a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_snprintf4() {
  char a[4] = {0};
  snprintf(a, sizeof(a), "%s/%s", a, a); // expected-warning{{Arguments must not be overlapping buffers}}
}

void test_snprintf5() {
  char a[4] = {0};
  snprintf(a, sizeof(a), "%d", 42); // no-warning
}

void test_snprintf6() {
  char a[4] = {0};
  char b[4] = {0};
  snprintf(a, sizeof(a), "%s", b); // no-warning
}


void memcpy(int dst, int src, size_t size); // expected-warning{{incompatible redeclaration of library function 'memcpy'}} expected-note{{'memcpy' is a builtin with type 'void *(void *, const void *, __size_t)' (aka 'void *(void *, const void *, unsigned long)')}}
void test_memcpy_proxy() {
  memcpy(42, 42, 42);
}

void strcpy(int dst, char *src); // expected-warning{{incompatible redeclaration of library function 'strcpy'}} expected-note{{'strcpy' is a builtin with type 'char *(char *, const char *)'}}
void test_strcpy_proxy() {
  strcpy(42, (char *)42);
}

void strxfrm(int dst, char *src, size_t size); // expected-warning{{incompatible redeclaration of library function 'strxfrm'}} expected-note{{'strxfrm' is a builtin with type '__size_t (char *, const char *, __size_t)' (aka 'unsigned long (char *, const char *, unsigned long)')}}
void test_strxfrm_proxy() {
  strxfrm(42, (char *)42, 42);
}
