// RUN: %clang_cl_asan /Od %s /Fe%t

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <malloc.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

// the size parameter is the size of the src buffer in bytes
// (dst has size 2 * size, for strcat reasons)
// the last two bytes are 0 (for str/wcs functions)
// they have distinct first bytes
// dst will be free'd if --fail is passed
typedef int test_function_t(void *, const void *, size_t);

#if __clang__
#  define DECLARE_WRAPPED(name) typeof(name) __asan_wrap_##name;
#else
// typeof is only supported in MSVC as of 17.7, with `-std:clatest`,
// and it doesn't seem to be possible to pass an option _only_ to cl
#  define DECLARE_WRAPPED(name) int __asan_wrap_##name();
#endif

#define TEST_FUNCTION_DECL(name)                                               \
  __declspec(dllexport) int test_##name(void *dst, const void *src, size_t size)
#define TEST_WRAPPED_FUNCTION_DECL(name)                                       \
  DECLARE_WRAPPED(name)                                                        \
  __declspec(dllexport) int test_wrap_##name(void *dst, const void *src,       \
                                             size_t size)

#define TEST_FUNCTION(name, ...)                                               \
  TEST_FUNCTION_DECL(name) { return 0 != name(__VA_ARGS__); }                  \
  TEST_WRAPPED_FUNCTION_DECL(name) {                                           \
    return 0 != __asan_wrap_##name(__VA_ARGS__);                               \
  }

#define TEST_NOT_FUNCTION(name, ...)                                           \
  TEST_FUNCTION_DECL(name) { return 0 == name(__VA_ARGS__); }                  \
  TEST_WRAPPED_FUNCTION_DECL(name) {                                           \
    return 0 == __asan_wrap_##name(__VA_ARGS__);                               \
  }

// RUN: %run %t --success memset 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped memset 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail memset 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped memset 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(memset, dst, *(const char *)src, size)
// RUN: %run %t --success memmove 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped memmove 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail memmove 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped memmove 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(memmove, dst, src, size)
// RUN: %run %t --success memcpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped memcpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail memcpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped memcpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(memcpy, dst, src, size)
// RUN: %run %t --success memcmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped memcmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail memcmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped memcmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(memcmp, dst, src, size)
// RUN: %run %t --success memchr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped memchr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail memchr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped memchr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_NOT_FUNCTION(memchr, dst, *(const char *)src, size)

// RUN: %run %t --success strlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strlen, dst)
// RUN: %run %t --success strnlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strnlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strnlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strnlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strnlen, dst, size)
// RUN: %run %t --success strcpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strcpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strcpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strcpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strcpy, dst, src)
// RUN: %run %t --success strncpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strncpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strncpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strncpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strncpy, dst, src, size)
// RUN: %run %t --success strcat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strcat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strcat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strcat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strcat, dst, src)
// RUN: %run %t --success strncat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strncat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strncat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strncat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strncat, dst, src, size)
// RUN: %run %t --success strcmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strcmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strcmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strcmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strcmp, dst, src)
// RUN: %run %t --success strncmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strncmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strncmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strncmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(strncmp, dst, src, size)
// RUN: %run %t --success strchr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped strchr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail strchr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped strchr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_NOT_FUNCTION(strchr, dst, *(const char *)src)

// RUN: %run %t --success wcslen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcslen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcslen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcslen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcslen, dst)
// RUN: %run %t --success wcsnlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcsnlen 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcsnlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcsnlen 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcsnlen, dst, size)
// RUN: %run %t --success wcscpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcscpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcscpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcscpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcscpy, dst, src)
// RUN: %run %t --success wcsncpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcsncpy 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcsncpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcsncpy 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcsncpy, dst, src, size / 2)
// RUN: %run %t --success wcscat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcscat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcscat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcscat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcscat, dst, src)
// RUN: %run %t --success wcsncat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcsncat 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcsncat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcsncat 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcsncat, dst, src, size / 2)
// RUN: %run %t --success wcscmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcscmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcscmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcscmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcscmp, dst, src)
// note: clang does not actually emit a call to wcsncmp, for some reason
// RUN: %run %t --success --wrapped wcsncmp 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail --wrapped wcsncmp 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_FUNCTION(wcsncmp, dst, src, size / 2)
// RUN: %run %t --success wcschr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %run %t --success --wrapped wcschr 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: not %run %t --fail wcschr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: not %run %t --fail --wrapped wcschr 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
TEST_NOT_FUNCTION(wcschr, dst, *(const wchar_t *)src)

void help(const char *progname) {
  fprintf(stderr, "Usage: %s (--fail|--success) [--wrapped] <intrinsic>\n",
          progname);
  exit(1);
}

enum SuccessOrFail {
  SOF_None,
  SOF_Succeed,
  SOF_Fail,
};

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    help(argv[0]);
  }

  const size_t size = 8;

  void *src = malloc(size);
  memset(src, 1, size);
  ((wchar_t *)src)[size / 2 - 1] = L'\0';

  void *dst = malloc(size * 2);
  memset(dst, 2, size);
  ((wchar_t *)dst)[size / 2 - 1] = L'\0';

  bool wrapped = false;
  enum SuccessOrFail succeed = SOF_None;
  const char *function_name = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--success") == 0) {
      succeed = SOF_Succeed;
    } else if (strcmp(argv[i], "--fail") == 0) {
      succeed = SOF_Fail;
    } else if (strcmp(argv[i], "--wrapped") == 0) {
      wrapped = true;
    } else {
      function_name = argv[i];
    }
  }

  if (!function_name || succeed == SOF_None) {
    help(argv[0]);
  } else if (succeed == SOF_Fail) {
    free(
        dst); // free dst, so that the function _should_ fail if ASan is correctly implemented
  }

  test_function_t *test_function = 0;
  char buffer[32];
  if (!wrapped) {
    strcpy(buffer, "test_");
  } else {
    strcpy(buffer, "test_wrap_");
  }

  // CHECK-FAIL: ERROR: AddressSanitizer: heap-use-after-free
  // CHECK-SUCCESS: pass
  // CHECK-SUCCESS-NOT: ERROR: AddressSanitizer: heap-use-after-free

  if (strncat(buffer, function_name, 32)) {
    test_function = (test_function_t *)GetProcAddress(0, buffer);
  }

  if (!test_function) {
    fprintf(stderr, "Unknown test: %s\n", argv[2]);
    return 1;
  }

  return !(test_function(dst, src, size) && printf("pass\n"));
}
