// RUN: %clangxx -O1 %s -o %t && %run %t
// UNSUPPORTED: android

// Fail on powerpc64 bots with:
// AddressSanitizer: CHECK failed: asan_thread.cpp:315 "((AddrIsInStack((uptr)&local))) != (0)"
// https://lab.llvm.org/buildbot/#/builders/18/builds/8162
// UNSUPPORTED: target=powerpc64{{.*}}
/// Occasionally fail on loongarch64 machine
// UNSUPPORTED: target=loongarch64{{.*}}

#include <assert.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (getenv("SANITIZER_TEST_REEXECED"))
    exit(0);
  struct rlimit rl;
  assert(!getrlimit(RLIMIT_STACK, &rl));
  struct rlimit rl_new = rl;
  rl_new.rlim_cur = 17351;
  assert(!setrlimit(RLIMIT_STACK, &rl_new));
  int pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    const char *envp[] = {"SANITIZER_TEST_REEXECED=1", nullptr};
    execve(argv[0], argv, const_cast<char **>(envp));
    assert(false);
  }
  int status;
  while (waitpid(-1, &status, __WALL) != pid) {
  }
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}
