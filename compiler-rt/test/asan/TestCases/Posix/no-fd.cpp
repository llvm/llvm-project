// RUN: %clangxx_asan -std=c++11 -O0 %s -o %t

// MallocNanoZone=0 disables initialization of the Nano MallocZone on Darwin.
// Initialization of this zone can interfere with this test because the zone
// might log which opens another file descriptor,
// e.g. failing to setup the zone due to ASan taking the memory region it wants.
// RUN: env MallocNanoZone=0 %run %t 2>&1 | FileCheck %s
// RUN: env MallocNanoZone=0 %env_asan_opts=debug=1,verbosity=2 %run %t 2>&1 | FileCheck %s

// Test ASan initialization

// This test closes the 0, 1, and 2 file descriptors before an exec() and relies
// on them remaining closed across an execve(). This is not the case on newer
// versions of Android. On PPC with ASLR turned on, this fails when linked with
// lld - see https://bugs.llvm.org/show_bug.cgi?id=45076.
// UNSUPPORTED: android, target=powerpc{{.*}}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" const char *__asan_default_options() {
  return "test_only_emulate_no_memorymap=1";
}

void parent(int argc, char **argv) {
  fprintf(stderr, "hello\n");
  // CHECK: hello
  close(0);
  close(1);
  dup2(2, 3);
  close(2);
  char *const newargv[] = {argv[0], (char *)"x", nullptr};
  execv(argv[0], newargv);
  perror("execve");
  exit(1);
}

void child() {
  assert(dup(3) == 0);
  assert(dup(3) == 1);
  assert(dup(3) == 2);
  fprintf(stderr, "world\n");
  // CHECK: world
}

int main(int argc, char **argv) {
  if (argc == 1) {
    parent(argc, argv);
  } else {
    child();
  }
}
