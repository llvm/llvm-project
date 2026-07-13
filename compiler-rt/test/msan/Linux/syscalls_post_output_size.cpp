// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// readlink/readlinkat return the byte count in res and do not NUL-terminate;
// getsockopt writes *optlen bytes of binary option data. The POST hooks used
// internal_strlen() to size the unpoison, which reads past what the kernel
// wrote. After the fix the unpoisoned span matches the real written length.

#include <assert.h>
#include <fcntl.h>
#include <string.h>
#include <sys/socket.h>

#include <sanitizer/linux_syscall_hooks.h>
#include <sanitizer/msan_interface.h>

int main() {
  {
    char buf[64];
    memset(buf, 'x', 64);
    buf[63] = 0; // in-bounds NUL so a stray strlen would run to 63, not OOB
    __msan_poison(buf, 64);
    __sanitizer_syscall_post_readlink(3, "/x", buf, 64);
    assert(__msan_test_shadow(buf, 64) == 3); // exactly res bytes
  }
  {
    char buf[64];
    memset(buf, 'x', 64);
    buf[63] = 0;
    __msan_poison(buf, 64);
    __sanitizer_syscall_post_readlinkat(3, AT_FDCWD, "/x", buf, 64);
    assert(__msan_test_shadow(buf, 64) == 3);
  }
  {
    unsigned char ov[16];
    __msan_poison(ov, 16);
    socklen_t ol = 4;
    __sanitizer_syscall_post_getsockopt(0, 3, 1, 2, ov, &ol);
    assert(__msan_test_shadow(ov, 16) == 4); // exactly *optlen bytes
  }
  return 0;
}
