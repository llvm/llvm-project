// Test fopencookie interceptor.
// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -std=c++11 -fsanitize-memory-track-origins -O0 %s -o %t && %run %t

// XFAIL: target-is-mips64el

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sanitizer/msan_interface.h>

constexpr uintptr_t kMagicCookie = 0x12345678;

static ssize_t cookie_read(void *cookie, char *buf, size_t size) {
  assert((uintptr_t)cookie == kMagicCookie);
  memset(buf, 0, size);
  return 0;
}

static ssize_t cookie_write(void *cookie, const char *buf, size_t size) {
  assert((uintptr_t)cookie == kMagicCookie);
  __msan_check_mem_is_initialized(buf, size);
  return 0;
}

static int cookie_seek(void *cookie, off64_t *offset, int whence) {
  assert((uintptr_t)cookie == kMagicCookie);
  __msan_check_mem_is_initialized(offset, sizeof(*offset));
  return 0;
}

static int cookie_close(void *cookie) {
  assert((uintptr_t)cookie == kMagicCookie);
  return 0;
}

void PoisonStack() { char a[8192]; }

void TestPoisonStack() {
  // Verify that PoisonStack has poisoned the stack - otherwise this test is not
  // testing anything.
  char a;
  assert(__msan_test_shadow(&a - 1000, 1) == 0);
}

int main() {
  void *cookie = (void *)kMagicCookie;
  FILE *f = fopencookie(cookie, "rw",
                        {cookie_read, cookie_write, cookie_seek, cookie_close});
  PoisonStack();
  TestPoisonStack();
  fseek(f, 100, SEEK_SET);
  char buf[50];
  fread(buf, 50, 1, f);
  fwrite(buf, 50, 1, f);
  fclose(f);

  f = fopencookie(cookie, "rw", {nullptr, nullptr, nullptr, nullptr});
  fseek(f, 100, SEEK_SET);
  fread(buf, 50, 1, f);
  fwrite(buf, 50, 1, f);
  fclose(f);
}
