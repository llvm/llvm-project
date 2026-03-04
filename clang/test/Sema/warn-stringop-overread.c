// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -verify -DUSE_BUILTINS
// RUN: %clang_cc1 -xc++ %s -verify
// RUN: %clang_cc1 -xc++ %s -verify -DUSE_BUILTINS

typedef unsigned long size_t;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(USE_BUILTINS)
#define memcpy(x,y,z) __builtin_memcpy(x,y,z)
#define memmove(x,y,z) __builtin_memmove(x,y,z)
#define memchr(x,y,z) __builtin_memchr(x,y,z)
#define memcmp(x,y,z) __builtin_memcmp(x,y,z)
#else
void *memcpy(void *dst, const void *src, size_t c);
void *memmove(void *dst, const void *src, size_t c);
void *memchr(const void *s, int c, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
#endif

int bcmp(const void *s1, const void *s2, size_t n);

#ifdef __cplusplus
}
#endif

void test_memcpy_overread(void) {
  char dst[100];
  int src = 0;
  memcpy(dst, &src, sizeof(src) + 1); // expected-warning {{'memcpy' reading 5 bytes from a region of size 4}}
}

void test_memcpy_array_overread(void) {
  int dest[10];
  int src[5] = {1, 2, 3, 4, 5};
  memcpy(dest, src, 10 * sizeof(int)); // expected-warning {{'memcpy' reading 40 bytes from a region of size 20}}
}

void test_memcpy_struct_overread(void) {
  struct S {
    int x;
    int y;
  };
  char dst[100];
  struct S src = {1, 2};
  memcpy(dst, &src, sizeof(struct S) + 1); // expected-warning {{'memcpy' reading 9 bytes from a region of size 8}}
}

void test_memmove_overread(void) {
  char dst[100];
  char src[10];
  memmove(dst, src, 20); // expected-warning {{'memmove' reading 20 bytes from a region of size 10}}
}

void test_memcpy_no_warning_exact_size(void) {
  char dst[100];
  int src = 0;
  memcpy(dst, &src, sizeof(src)); // no warning
}

void test_memcpy_no_warning_smaller_size(void) {
  char dst[100];
  int src[10];
  memcpy(dst, src, 5 * sizeof(int)); // no warning
}

void test_memcpy_both_overflow(void) {
  char dst[5];
  int src = 0;
  memcpy(dst, &src, 10); // expected-warning {{'memcpy' reading 10 bytes from a region of size 4}}
                         // expected-warning@-1 {{'memcpy' will always overflow; destination buffer has size 5, but size argument is 10}}
}

void test_memchr_overread(void) {
  char buf[4];
  memchr(buf, 'a', 8); // expected-warning {{'memchr' reading 8 bytes from a region of size 4}}
}

void test_memchr_no_warning(void) {
  char buf[10];
  memchr(buf, 'a', 10); // no warning
}

void test_memcmp_overread_first(void) {
  char a[4], b[100];
  memcmp(a, b, 8); // expected-warning {{'memcmp' reading 8 bytes from a region of size 4}}
}

void test_memcmp_overread_second(void) {
  char a[100], b[4];
  memcmp(a, b, 8); // expected-warning {{'memcmp' reading 8 bytes from a region of size 4}}
}

void test_memcmp_overread_both(void) {
  char a[4], b[2];
  memcmp(a, b, 8); // expected-warning {{'memcmp' reading 8 bytes from a region of size 4}} \
                    // expected-warning {{'memcmp' reading 8 bytes from a region of size 2}}
}

void test_memcmp_no_warning(void) {
  char a[10], b[10];
  memcmp(a, b, 10); // no warning
}

void test_memcpy_src_offset_overread(void) {
  char src[] = {1, 2, 3, 4};
  char dst[10];
  memcpy(dst, src + 2, 3); // expected-warning {{'memcpy' reading 3 bytes from a region of size 2}}
}

void test_memcpy_src_offset_no_warning(void) {
  char src[] = {1, 2, 3, 4};
  char dst[10];
  memcpy(dst, src + 2, 2); // no warning
}

void test_bcmp_overread(void) {
  char a[4], b[100];
  bcmp(a, b, 8); // expected-warning {{'bcmp' reading 8 bytes from a region of size 4}}
}

void test_bcmp_no_warning(void) {
  char a[10], b[10];
  bcmp(a, b, 10); // no warning
}

void test_memcpy_chk_overread(void) {
  char dst[100];
  char src[4];
  __builtin___memcpy_chk(dst, src, 8, sizeof(dst)); // expected-warning {{'memcpy' reading 8 bytes from a region of size 4}}
}

void test_memmove_chk_overread(void) {
  char dst[100];
  char src[4];
  __builtin___memmove_chk(dst, src, 8, sizeof(dst)); // expected-warning {{'memmove' reading 8 bytes from a region of size 4}}
}

#ifdef __cplusplus
template <int N>
void test_memcpy_dependent_dest() {
  char dst[N];
  int src = 0;
  memcpy(dst, &src, sizeof(src) + 1); // expected-warning {{'memcpy' reading 5 bytes from a region of size 4}}
}

void call_test_memcpy_dependent_dest() {
  test_memcpy_dependent_dest<100>(); // expected-note {{in instantiation}}
}

// FIXME: We should warn here at the template definition since src and size are
// not dependent, but checkFortifiedBuiltinMemoryFunction exits when any part of
// the call is dependent (and thus uninstantiated).
template <int N>
void test_memcpy_dependent_dest_uninstantiated() {
  char dst[N];
  int src = 0;
  memcpy(dst, &src, sizeof(src) + 1); // missing-warning {{'memcpy' reading 5 bytes from a region of size 4}}
}

#endif
