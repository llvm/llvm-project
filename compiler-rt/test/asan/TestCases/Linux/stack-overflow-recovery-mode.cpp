// Test that ASan doesn't hang on stack overflow in recovery mode.
//
// RUN: %clang_asan -O0 -fsanitize-recover=address %s -o %t
// RUN: %env_asan_opts=halt_on_error=false not %run %t 2>&1 | FileCheck %s

// Issue #109771
// XFAIL: target={{sparc.*-.*-linux.*}}

// Waiting for https://github.com/llvm/llvm-project/pull/236879: libunwind skips
// CFI instructions at the faulting PC in signal frames, causing SIGILL on
// AArch64 PAC-RET prologues under FEAT_FPAC.
// UNSUPPORTED: android && aarch64-target-arch

#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>

static volatile int *recurse(volatile int n, volatile int *p) {
  // CHECK: {{stack-overflow on address 0x.* \(pc 0x.* bp 0x.* sp 0x.* T.*\)}}
  if (n >= 0) *recurse(n + 1, p) += n;
  return p;
}


void LimitStackAndReexec(int argc, char **argv) {
  struct rlimit rlim;
  int res = getrlimit(RLIMIT_STACK, &rlim);
  assert(res == 0);
  if (rlim.rlim_cur == RLIM_INFINITY) {
    rlim.rlim_cur = 256 * 1024;
    res = setrlimit(RLIMIT_STACK, &rlim);
    assert(res == 0);

    execv(argv[0], argv);
    assert(0 && "unreachable");
  }
}

int main(int argc, char **argv) {
  LimitStackAndReexec(argc, argv);
  volatile int res;
  return *recurse(argc + 1, &res);
}
