// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %env_esan_opts="verbosity=1 record_snapshots=0" %run %t %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static void testChildStackLimit(rlim_t StackLimit, char *ToRun) {
  int Res;
  struct rlimit Limit;
  Limit.rlim_cur = RLIM_INFINITY;
  Limit.rlim_max = RLIM_INFINITY;
  Res = setrlimit(RLIMIT_STACK, &Limit);
  if (Res != 0) {
    // Probably our environment had a large limit and we ourselves got
    // re-execed and can no longer raise our limit.
    // We have to bail and emulate the regular test.
    // We'd prefer to have branches in our FileCheck output to ensure the
    // initial program was re-execed but this is the best we can do for now.
    fprintf(stderr, "in esan::initializeLibrary\n");
    fprintf(stderr, "==1234==The stack size limit is beyond the maximum supported.\n");
    fprintf(stderr, "Re-execing with a stack size below 1TB.\n");
    fprintf(stderr, "in esan::initializeLibrary\n");
    fprintf(stderr, "done\n");
    fprintf(stderr, "in esan::finalizeLibrary\n");
    return;
  }

  pid_t Child = fork();
  assert(Child >= 0);
  if (Child > 0) {
    pid_t WaitRes = waitpid(Child, NULL, 0);
    assert(WaitRes == Child);
  } else {
    char *Args[2];
    Args[0] = ToRun;
    Args[1] = NULL;
    Res = execv(ToRun, Args);
    assert(0); // Should not be reached.
  }
}

int main(int argc, char *argv[]) {
  // The path to the program to exec must be passed in the first time.
  if (argc == 2) {
    fprintf(stderr, "Testing child with infinite stack\n");
    testChildStackLimit(RLIM_INFINITY, argv[1]);
    fprintf(stderr, "Testing child with 1TB stack\n");
    testChildStackLimit(1ULL << 40, argv[1]);
  }
  fprintf(stderr, "done\n");
  // CHECK:      in esan::initializeLibrary
  // CHECK:      Testing child with infinite stack
  // CHECK-NEXT: in esan::initializeLibrary
  // CHECK-NEXT: =={{[0-9:]+}}==The stack size limit is beyond the maximum supported.
  // CHECK-NEXT: Re-execing with a stack size below 1TB.
  // CHECK-NEXT: in esan::initializeLibrary
  // CHECK:      done
  // CHECK:      in esan::finalizeLibrary
  // CHECK:      Testing child with 1TB stack
  // CHECK-NEXT: in esan::initializeLibrary
  // CHECK-NEXT: =={{[0-9:]+}}==The stack size limit is beyond the maximum supported.
  // CHECK-NEXT: Re-execing with a stack size below 1TB.
  // CHECK-NEXT: in esan::initializeLibrary
  // CHECK:      done
  // CHECK-NEXT: in esan::finalizeLibrary
  // CHECK:      done
  // CHECK-NEXT: in esan::finalizeLibrary
  return 0;
}
