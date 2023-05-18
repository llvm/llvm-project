// Regression test for https://crbug.com/502974, where ASan was unable to read
// the binary name because of sandbox restrictions.
// This test uses seccomp-BPF to restrict the readlink() system call and makes
// sure ASan is still able to
// Disable symbolizing results, since this will invoke llvm-symbolizer, which
// will be unable to resolve its $ORIGIN due to readlink() restriction and will
// thus fail to start, causing the test to die with SIGPIPE when attempting to
// talk to it.
// RUN: not ls /usr/include/linux/seccomp.h || ( %clang_asan %s -o %t && ( not env ASAN_OPTIONS=symbolize=0 %run %t 2>&1 ) | FileCheck %s )
// REQUIRES: shell
// UNSUPPORTED: android

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

#ifndef __NR_readlink
# define __NR_readlink __NR_readlinkat
#endif

#define syscall_nr (offsetof(struct seccomp_data, nr))

void corrupt() {
  void *p = malloc(10);
  free(p);
  free(p);
}

int main() {
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

  struct sock_filter filter[] = {
    /* Grab the system call number */
    BPF_STMT(BPF_LD + BPF_W + BPF_ABS, syscall_nr),
    // If this is __NR_readlink,
    BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, __NR_readlink, 0, 1),
    // return with EPERM,
    BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_ERRNO | EPERM),
    // otherwise allow the syscall.
    BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_ALLOW)
  };
  struct sock_fprog prog;
  prog.len = (unsigned short)(sizeof(filter)/sizeof(filter[0]));
  prog.filter = filter;

  int res = prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog, 0, 0);
  if (res != 0) {
    fprintf(stderr, "PR_SET_SECCOMP unsupported!\n");
  }
  corrupt();
  // CHECK: AddressSanitizer
  // CHECK-NOT: reading executable name failed
  return 0;
}
