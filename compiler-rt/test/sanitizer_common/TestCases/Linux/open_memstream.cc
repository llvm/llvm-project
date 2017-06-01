// RUN: %clangxx -m64 -O0 -g -xc++ %s -o %t && %run %t
// RUN: %clangxx -m64 -O3 -g -xc++ %s -o %t && %run %t
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
static void check_mem_is_good(void *p, size_t s) {
  __msan_check_mem_is_initialized(p, s);
}
#elif __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
static void check_mem_is_good(void *p, size_t s) {
  assert(__asan_region_is_poisoned(p, s) == 0);
}
#else
static void check_mem_is_good(void *p, size_t s) {}
#endif

static void run(bool flush) {
  char *buf;
  size_t buf_len;
  fprintf(stderr, " &buf %p, &buf_len %p\n", &buf, &buf_len);
  FILE *fp = open_memstream(&buf, &buf_len);
  fprintf(fp, "hello");
  if (flush) {
    fflush(fp);
    check_mem_is_good(&buf, sizeof(buf));
    check_mem_is_good(&buf_len, sizeof(buf_len));
    check_mem_is_good(buf, buf_len);
  }

  char *p = new char[1024];
  memset(p, 'a', 1023);
  p[1023] = 0;
  for (int i = 0; i < 100; ++i)
    fprintf(fp, "%s", p);
  delete[] p;

  if (flush) {
    fflush(fp);
    fprintf(stderr, " %p addr %p, len %zu\n", &buf, buf, buf_len);
    check_mem_is_good(&buf, sizeof(buf));
    check_mem_is_good(&buf_len, sizeof(buf_len));
    check_mem_is_good(buf, buf_len);\
  }

  fclose(fp);
  check_mem_is_good(&buf, sizeof(buf));
  check_mem_is_good(&buf_len, sizeof(buf_len));
  check_mem_is_good(buf, buf_len);

  free(buf);
}

int main(void) {
  for (int i = 0; i < 100; ++i)
    run(false);
  for (int i = 0; i < 100; ++i)
    run(true);
  return 0;
}
