// RUN: %clangxx_msan -O0 %s -o %t -lresolv && %run %t
// RUN: not %run %t NTOP_READ 2>&1 | FileCheck %s --check-prefix=NTOP_READ
// RUN: not %run %t PTON_READ 2>&1 | FileCheck %s --check-prefix=PTON_READ

#include <assert.h>
#include <resolv.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <sanitizer/msan_interface.h>

int main(int iArgc, char *szArgv[]) {
  char* test = nullptr;
  if (iArgc > 1) {
    test = szArgv[1];
  }

  if (test == nullptr) {
    // Check NTOP writing
    const char *src = "base64 test data";
    size_t src_len = strlen(src);
    size_t dst_len = ((src_len + 2) / 3) * 4 + 1;
    char dst[dst_len];
    int res = b64_ntop(reinterpret_cast<const unsigned char *>(src), src_len,
                       dst, dst_len);
    assert(res >= 0);
    __msan_check_mem_is_initialized(dst, res);

    // Check PTON writing
    unsigned char target[dst_len];
    res = b64_pton(dst, target, dst_len);
    assert(res >= 0);
    __msan_check_mem_is_initialized(target, res);

    // Check NTOP writing for zero length src
    src = "";
    src_len = strlen(src);
    assert(((src_len + 2) / 3) * 4 + 1 < dst_len);
    res = b64_ntop(reinterpret_cast<const unsigned char *>(src), src_len,
                       dst, dst_len);
    assert(res >= 0);
    __msan_check_mem_is_initialized(dst, res);

    // Check PTON writing for zero length src
    dst[0] = '\0';
    res = b64_pton(dst, target, dst_len);
    assert(res >= 0);
    __msan_check_mem_is_initialized(target, res);

    return 0;
  }

  if (strcmp(test, "NTOP_READ") == 0) {
    // Check NTOP reading
    size_t src_len = 80;
    char src[src_len];
    __msan_poison(src, src_len);
    size_t dst_len = ((src_len + 2) / 3) * 4 + 1;
    char dst[dst_len];
    int res = b64_ntop(reinterpret_cast<const unsigned char *>(src), src_len,
                       dst, dst_len);
    // NTOP_READ: Uninitialized bytes in __interceptor___b64_ntop
    return 0;
  }

  if (strcmp(test, "PTON_READ") == 0) {
    // Check PTON reading
    size_t src_len = 80;
    char src[src_len];
    strcpy(src, "junk longer than zero");
    // pretend it is uninitialized
    __msan_poison(src, src_len);
    unsigned char target[src_len];
    int res = b64_pton(src, target, src_len);
    // PTON_READ: Uninitialized bytes in __interceptor___b64_pton
    return 0;
  }

  return 0;
}
