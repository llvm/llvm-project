// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test whether calling certain libFuzzer's interceptors inside allocators
// does not cause an assertion failure.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <malloc.h>

static const char *buf1 = "aaaa";
static const char *buf2 = "bbbb";

static void callFuzzerInterceptors(const char *prefix) {
  int memcmp_result = memcmp(buf1, buf2, 4);
  if (memcmp_result != 0) {
    fprintf(stderr, "%s-MEMCMP\n", prefix);
  }
  int strncmp_result = strncmp(buf1, buf2, 4);
  if (strncmp_result != 0) {
    fprintf(stderr, "%s-STRNCMP\n", prefix);
  }
  int strcmp_result = strcmp(buf1, buf2);
  if (strcmp_result != 0) {
    fprintf(stderr, "%s-STRCMP\n", prefix);
  }
  const char *strstr_result = strstr(buf1, buf2);
  if (strstr_result == nullptr) {
    fprintf(stderr, "%s-STRSTR\n", prefix);
  }
}

extern "C" void *__libc_calloc(size_t, size_t);

extern "C" void *calloc(size_t n, size_t elem_size) {
  static bool CalledOnce = false;
  if (!CalledOnce) {
    callFuzzerInterceptors("CALLOC");
    CalledOnce = true;
  }
  return __libc_calloc(n, elem_size);
}
