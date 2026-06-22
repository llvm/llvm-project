// RUN: %clang_cc1 -fsyntax-only -Wfortify-source -verify %s

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *memcpy(void *, const void *, size_t);
size_t strlen(const char *);

void direct_malloc_strlen(const char *src) {
  char *p = malloc(strlen(src)); // expected-warning{{allocation size does not include space for null terminator; consider 'strlen(src) + 1'}}
  char *ok = malloc(strlen(src) + 1);
  (void)p;
  (void)ok;
}

void direct_memcpy_strlen(char *dst, const char *src) {
  memcpy(dst, src, strlen(src)); // expected-warning{{copy size does not include space for null terminator; consider 'strlen(src) + 1'}}
  memcpy(dst, src, strlen(src) + 1);
}
