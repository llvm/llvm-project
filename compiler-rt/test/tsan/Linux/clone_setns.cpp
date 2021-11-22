// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// The test models how sandbox2 unshares user namespace after clone:
// https://github.com/google/sandboxed-api/blob/c95837a6c131fbdf820db352a97d54fcbcbde6c0/sandboxed_api/sandbox2/forkserver.cc#L249
// which works only in sigle-threaded processes.

#include "../test.h"
#include <errno.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>

static int cloned(void *arg) {
  // Unshare can fail for other reasons, e.g. no permissions,
  // so check only the error we are interested in:
  // if the process is multi-threaded unshare must return EINVAL.
  if (unshare(CLONE_NEWUSER) && errno == EINVAL) {
    fprintf(stderr, "unshare failed: %d\n", errno);
    exit(1);
  }
  exit(0);
  return 0;
}

int main() {
  char stack[64 << 10] __attribute__((aligned(64)));
  int pid = clone(cloned, stack + sizeof(stack), SIGCHLD, 0);
  if (pid == -1) {
    fprintf(stderr, "failed to clone: %d\n", errno);
    exit(1);
  }
  int status = 0;
  while (wait(&status) != pid) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "child failed: %d\n", status);
    exit(1);
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
