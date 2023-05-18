// RUN: %clang %s -o %t && %run %t
// capget() and capset() are not intercepted on Android.
// UNSUPPORTED: android

#include <assert.h>
#include <errno.h>
#include <linux/capability.h>
#include <stdio.h>
#include <stdlib.h>

#include "sanitizer_common/sanitizer_specific.h"

/* Use capget() and capset() from glibc. */
int capget(cap_user_header_t header, cap_user_data_t data);
int capset(cap_user_header_t header, const cap_user_data_t data);

static void test(int version, int u32s) {
  struct __user_cap_header_struct hdr = {
      .version = version,
      .pid = 0,
  };
  struct __user_cap_data_struct data[u32s];
  if (capget(&hdr, data)) {
    assert(errno == EINVAL);
    /* Check that memory is not touched. */
#if __has_feature(memory_sanitizer)
    assert(__msan_test_shadow(data, sizeof(data)) == 0);
#endif
    hdr.version = version;
    int err = capset(&hdr, data);
    assert(errno == EINVAL);
  } else {
    for (int i = 0; i < u32s; i++)
      printf("%x %x %x\n", data[i].effective, data[i].permitted,
             data[i].inheritable);
    int err = capset(&hdr, data);
    assert(!err);
  }
}

int main() {
  test(0, 1); /* Test an incorrect version. */
  test(_LINUX_CAPABILITY_VERSION_1, _LINUX_CAPABILITY_U32S_1);
  test(_LINUX_CAPABILITY_VERSION_2, _LINUX_CAPABILITY_U32S_2);
  test(_LINUX_CAPABILITY_VERSION_3, _LINUX_CAPABILITY_U32S_3);

  return EXIT_SUCCESS;
}
