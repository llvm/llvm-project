// Test the weak hooks.
// RUN: %clangxx %s -o %t
// RUN: %run %t

// Hooks are not implemented for lsan.
// XFAIL: lsan
// XFAIL: ubsan

#include <assert.h>
#include <string.h>
#if defined(_GNU_SOURCE)
#include <strings.h> // for bcmp
#endif

bool seen_memcmp, seen_strncmp, seen_strncasecmp, seen_strcmp, seen_strcasecmp,
    seen_strstr, seen_strcasestr, seen_memmem;

extern "C" {
void __sanitizer_weak_hook_memcmp(void *called_pc, const void *s1,
                                  const void *s2, size_t n, int result) {
  seen_memcmp = true;
}
void __sanitizer_weak_hook_strncmp(void *called_pc, const char *s1,
                                   const char *s2, size_t n, int result) {
  seen_strncmp = true;
}
void __sanitizer_weak_hook_strncasecmp(void *called_pc, const char *s1,
                                       const char *s2, size_t n, int result){
  seen_strncasecmp = true;
}
void __sanitizer_weak_hook_strcmp(void *called_pc, const char *s1,
                                  const char *s2, int result){
  seen_strcmp = true;
}
void __sanitizer_weak_hook_strcasecmp(void *called_pc, const char *s1,
                                      const char *s2, int result){
  seen_strcasecmp = true;
}
void __sanitizer_weak_hook_strstr(void *called_pc, const char *s1,
                                  const char *s2, char *result){
  seen_strstr = true;
}
void __sanitizer_weak_hook_strcasestr(void *called_pc, const char *s1,
                                      const char *s2, char *result){
  seen_strcasestr = true;
}
void __sanitizer_weak_hook_memmem(void *called_pc, const void *s1, size_t len1,
                                  const void *s2, size_t len2, void *result){
  seen_memmem = true;
}
} // extern "C"

char s1[] = "ABCDEF";
char s2[] = "CDE";

static volatile int int_sink;
static volatile void *ptr_sink;

int main() {
  assert(sizeof(s2) < sizeof(s1));

  int_sink = memcmp(s1, s2, sizeof(s2));
  assert(seen_memcmp);

#if defined(_GNU_SOURCE) || defined(__NetBSD__) || defined(__FreeBSD__) || \
    defined(__OpenBSD__)
  seen_memcmp = false;
  int_sink = bcmp(s1, s2, sizeof(s2));
  assert(seen_memcmp);
#endif

  int_sink = strncmp(s1, s2, sizeof(s2));
  assert(seen_strncmp);

  int_sink = strncasecmp(s1, s2, sizeof(s2));
  assert(seen_strncasecmp);

  int_sink = strcmp(s1, s2);
  assert(seen_strcmp);

  int_sink = strcasecmp(s1, s2);
  assert(seen_strcasecmp);

  ptr_sink = strstr(s1, s2);
  assert(seen_strstr);

  ptr_sink = strcasestr(s1, s2);
  assert(seen_strcasestr);

  ptr_sink = memmem(s1, sizeof(s1), s2, sizeof(s2));
  assert(seen_memmem);
  return 0;
}
