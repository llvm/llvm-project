// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=debug=1,verbosity=2 %run %t 2>&1 | FileCheck %s

// Test ASan initialization

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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
