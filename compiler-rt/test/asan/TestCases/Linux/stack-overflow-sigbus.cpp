// Test ASan detection of stack-overflow condition when Linux sends SIGBUS.

// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s

// Issue #109771
// XFAIL: target={{sparc.*-.*-linux.*}}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>

const int BS = 1024;
volatile char x;
volatile int y = 1;

void recursive_func(char *p) {
  char buf[BS];
  buf[rand() % BS] = 1;
  buf[rand() % BS] = 2;
  x = buf[rand() % BS];
  if (y)
    recursive_func(buf);
  x = 1; // prevent tail call optimization
  // CHECK: {{stack-overflow on address 0x.* \(pc 0x.* bp 0x.* sp 0x.* T.*\)}}
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

  // Map some memory just before the start of the current stack vma.
  // When the stack grows down and crashes into it, Linux can send
  // SIGBUS instead of SIGSEGV. See:
  // http://lkml.iu.edu/hypermail/linux/kernel/1008.1/02299.html
  const long pagesize = sysconf(_SC_PAGESIZE);
  FILE *f = fopen("/proc/self/maps", "r");
  char a[1000];
  void *p = 0;
  while (fgets(a, sizeof a, f)) {
    if (strstr(a, "[stack]")) {
      unsigned long addr;
      if (sscanf(a, "%lx", &addr) == 1)
        p = mmap((void *)(addr - 4 * pagesize), pagesize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
  }
  assert(p);

  recursive_func(0);
  return 0;
}
