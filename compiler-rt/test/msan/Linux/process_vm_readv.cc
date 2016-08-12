// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t -DPOSITIVE && not %run %t |& FileCheck %s

// XFAIL: target-is-mips64el

#include <assert.h>
#include <dlfcn.h>
#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <errno.h>

typedef ssize_t (*process_vm_readwritev_fn)(pid_t, const iovec *, unsigned long,
                                            const iovec *, unsigned long,
                                            unsigned long);

// Exit with success, emulating the expected output.
int exit_dummy()
{
#ifdef POSITIVE
    printf("process_vm_readv not found or not implemented!\n");
    printf(
        "WARNING: MemorySanitizer: use-of-uninitialized-value (not really)\n");
    return 1;
#else
    return 0;
#endif
}

int main(void) {
  // This requires glibc 2.15.
  process_vm_readwritev_fn libc_process_vm_readv =
      (process_vm_readwritev_fn)dlsym(RTLD_NEXT, "process_vm_readv");
  if (!libc_process_vm_readv)
    return exit_dummy();

  process_vm_readwritev_fn process_vm_readv =
      (process_vm_readwritev_fn)dlsym(RTLD_DEFAULT, "process_vm_readv");
  process_vm_readwritev_fn process_vm_writev =
      (process_vm_readwritev_fn)dlsym(RTLD_DEFAULT, "process_vm_writev");

  char a[100];
  memset(a, 0xab, 100);

  char b[100];
  iovec iov_a[] = {{(void *)a, 20}, (void *)(a + 50), 10};
  iovec iov_b[] = {{(void *)(b + 10), 10}, (void *)(b + 30), 20};

  __msan_poison(&b, sizeof(b));
  ssize_t res = process_vm_readv(getpid(), iov_b, 2, iov_a, 2, 0);
  if (errno == ENOSYS) // Function not implemented 
    return exit_dummy();

  assert(res == 30);
  __msan_check_mem_is_initialized(b + 10, 10);
  __msan_check_mem_is_initialized(b + 30, 20);
  assert(__msan_test_shadow(b + 9, 1) == 0);
  assert(__msan_test_shadow(b + 20, 1) == 0);
  assert(__msan_test_shadow(b + 29, 1) == 0);
  assert(__msan_test_shadow(b + 50, 1) == 0);

#ifdef POSITIVE
  __msan_unpoison(&b, sizeof(b));
  __msan_poison(b + 32, 1);
  res = process_vm_writev(getpid(), iov_b, 2, iov_a, 2, 0);
// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
#else
  __msan_unpoison(&b, sizeof(b));
  res = process_vm_writev(getpid(), iov_b, 2, iov_a, 2, 0);
  assert(res == 30);
#endif

  return 0;
}
