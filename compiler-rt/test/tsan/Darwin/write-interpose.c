// Test that dylibs interposing write, and then calling functions intercepted
// by TSan don't deadlock (self-lock)

// RUN: %clang_tsan %s -o %t
// RUN: %clang_tsan %s -o %t.dylib -fno-sanitize=thread -dynamiclib -DSHARED_LIB

// Note that running the below command with out `lock_during_write` should
// deadlock (self-lock)
// RUN: env DYLD_INSERT_LIBRARIES=%t.dylib TSAN_OPTIONS=verbosity=2:lock_during_write=disable_for_current_process %run %t 2>&1 | FileCheck %s

#include <stdio.h>

#if defined(SHARED_LIB)

// dylib implementation - interposes write() calls
#  include <os/lock.h>
#  include <unistd.h>

struct interpose_substitution {
  const void *replacement;
  const void *original;
};

#  define INTERPOSE(replacement, original)                                     \
    __attribute__((used)) static const struct interpose_substitution           \
        substitution_##original[]                                              \
        __attribute__((section("__DATA, __interpose"))) = {                    \
            {(const void *)(replacement), (const void *)(original)}}

static ssize_t my_write(int fd, const void *buf, size_t count) {
  struct os_unfair_lock_s lock = OS_UNFAIR_LOCK_INIT;
  os_unfair_lock_lock(&lock);
  printf("Interposed write called: fd=%d, count=%zu\n", fd, count);
  ssize_t res = write(fd, buf, count);
  os_unfair_lock_unlock(&lock);
  return res;
}
INTERPOSE(my_write, write);

#else // defined(SHARED_LIB)

int main() {
  printf("Write test completed\n");
  return 0;
}

#endif // defined(SHARED_LIB)

// CHECK: Interposed write called: fd={{[0-9]+}}, count={{[0-9]+}}
// CHECK: Write test completed
