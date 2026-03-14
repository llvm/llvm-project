// RUN: %clang_cc1 -fsyntax-only -verify %s
//

typedef __SIZE_TYPE__ size_t;
extern "C" void *memset(void *, int, size_t);
extern "C" void *memmove(void *s1, const void *s2, size_t n);
extern "C" void *memcpy(void *s1, const void *s2, size_t n);
extern "C" int memcmp(void *s1, const void *s2, size_t n);
extern "C" int strncmp(const char *s1, const char *s2, size_t n);
extern "C" int strncasecmp(const char *s1, const char *s2, size_t n);
extern "C" char *strncpy(char *dst, const char *src, size_t n);
extern "C" char *strncat(char *dst, const char *src, size_t n);
extern "C" char *strndup(const  char *src, size_t n);
extern "C" size_t strlcpy(char *dst, const char *src, size_t size);
extern "C" size_t strlcat(char *dst, const char *src, size_t size);

void f() {
  char b1[80], b2[80];
  if (memset(b1, 0, sizeof(b1) != 0)) {} // \
    expected-warning{{size argument in 'memset' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (memset(b1, 0, sizeof(b1)) != 0) {}

  if (memmove(b1, b2, sizeof(b1) == 0)) {} // \
    expected-warning{{size argument in 'memmove' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (memmove(b1, b2, sizeof(b1)) == 0) {}

  // FIXME: This fixit is bogus.
  if (memcpy(b1, b2, sizeof(b1) < 0)) {} // \
    expected-warning{{size argument in 'memcpy' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (memcpy(b1, b2, sizeof(b1)) < 0) {} // expected-error {{ordered comparison between pointer and zero}}

  if (memcmp(b1, b2, sizeof(b1) <= 0)) {} // \
    expected-warning{{size argument in 'memcmp' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (memcmp(b1, b2, sizeof(b1)) <= 0) {}

  if (strncmp(b1, b2, sizeof(b1) > 0)) {} // \
    expected-warning{{size argument in 'strncmp' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strncmp(b1, b2, sizeof(b1)) > 0) {}

  if (strncasecmp(b1, b2, sizeof(b1) >= 0)) {} // \
    expected-warning{{size argument in 'strncasecmp' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strncasecmp(b1, b2, sizeof(b1)) >= 0) {}

  if (strncpy(b1, b2, sizeof(b1) == 0 || true)) {} // \
    expected-warning{{size argument in 'strncpy' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strncpy(b1, b2, sizeof(b1)) == 0 || true) {}

  // FIXME: This fixit is bogus.
  if (strncat(b1, b2, sizeof(b1) - 1 >= 0 && true)) {} // \
    expected-warning{{size argument in 'strncat' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strncat(b1, b2, sizeof(b1) - 1) >= 0 && true) {} // expected-error {{ordered comparison between pointer and zero}}

  if (strndup(b1, sizeof(b1) != 0)) {} // \
    expected-warning{{size argument in 'strndup' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strndup(b1, sizeof(b1)) != 0) {}

  if (strlcpy(b1, b2, sizeof(b1) != 0)) {} // \
    expected-warning{{size argument in 'strlcpy' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strlcpy(b1, b2, sizeof(b1)) != 0) {}

  if (strlcat(b1, b2, sizeof(b1) != 0)) {} // \
    expected-warning{{size argument in 'strlcat' call is a comparison}} \
    expected-note {{did you mean to compare}} \
    expected-note {{explicitly cast the argument}}
  if (strlcat(b1, b2, sizeof(b1)) != 0) {}

  if (memset(b1, 0, sizeof(b1) / 2)) {}
  if (memset(b1, 0, sizeof(b1) >> 2)) {}
  if (memset(b1, 0, 4 << 2)) {}
  if (memset(b1, 0, 4 + 2)) {}
  if (memset(b1, 0, 4 - 2)) {}
  if (memset(b1, 0, 4 * 2)) {}

  if (memset(b1, 0, (size_t)(sizeof(b1) != 0))) {}
}
