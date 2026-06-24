// RUN: %clang_cc1 -fsyntax-only -Wfortify-source -verify %s

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *realloc(void *, size_t);
void *memcpy(void *, const void *, size_t);
char *strcpy(char *, const char *);
size_t strlen(const char *);

void malloc_len_then_strcpy(const char *src, size_t len) {
  char *s = malloc(len); // expected-warning{{allocation of 'len' bytes may be insufficient for null-terminated string; consider 'len + 1'}}
  strcpy(s, src);
}

void malloc_len_non_string(char *dst, const char *src, size_t len) {
  char *buf = malloc(len);
  memcpy(buf, src, len);
  memcpy(dst, buf, len);
}

void malloc_complex_size(const char *src, size_t len) {
  char *s = malloc(len * 2);
  strcpy(s, src);
}

void realloc_len_then_strcpy(char *s, const char *src, size_t len) {
  s = realloc(s, len); // expected-warning{{realloc may remove space required for null terminator; expected at least len + 1}}
  strcpy(s, src);
}

void realloc_ok(char *s, const char *src, size_t len) {
  s = realloc(s, len + 1);
  strcpy(s, src);
}

void malloc_len_then_memcpy_string(const char *src, size_t len) {
  char *s = malloc(len); // expected-warning{{allocation of 'len' bytes may be insufficient for null-terminated string; consider 'len + 1'}}
  memcpy(s, src, strlen(src) + 1);
}
