// RUN: %clang_scudo %s -o %t
// RUN: %run %t after  2>&1 | FileCheck %s
// RUN: %run %t before 2>&1 | FileCheck %s

// Test that we hit a guard page when writing past the end of a chunk
// allocated by the Secondary allocator, or writing too far in front of it.

#include <assert.h>
#include <malloc.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void handler(int signo, siginfo_t *info, void *uctx) {
  if (info->si_code == SEGV_ACCERR) {
    fprintf(stderr, "SCUDO SIGSEGV\n");
    exit(0);
  }
  exit(1);
}

int main(int argc, char **argv) {
  // The size must be large enough to be serviced by the secondary allocator.
  long page_size = sysconf(_SC_PAGESIZE);
  size_t size = (1U << 17) + page_size;
  struct sigaction a;

  assert(argc == 2);
  memset(&a, 0, sizeof(a));
  a.sa_sigaction = handler;
  a.sa_flags = SA_SIGINFO;

  char *p = (char *)malloc(size);
  assert(p);
  memset(p, 'A', size); // This should not trigger anything.
  // Set up the SIGSEGV handler now, as the rest should trigger an AV.
  sigaction(SIGSEGV, &a, NULL);
  if (!strcmp(argv[1], "after")) {
    for (int i = 0; i < page_size; i++)
      p[size + i] = 'A';
  }
  if (!strcmp(argv[1], "before")) {
    for (int i = 1; i < page_size; i++)
      p[-i] = 'A';
  }
  free(p);

  return 1; // A successful test means we shouldn't reach this.
}

// CHECK: SCUDO SIGSEGV
