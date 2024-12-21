// RUN: %clangxx -fsanitize=realtime -DIS_NONBLOCKING=1 %s -o %t
// RUN: %env_rtsan_opts="halt_on_error=true" not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-HALT
// RUN: %env_rtsan_opts="halt_on_error=false" %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOHALT

// RUN: %clangxx -fsanitize=realtime -DIS_NONBLOCKING=0 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OK
// RUN: %env_rtsan_opts="halt_on_error=false" %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OK

// UNSUPPORTED: ios

// Intent: Ensure fork/exec dies when realtime and survives otherwise
//         This behavior is difficult to test in a gtest, because the process is
//         wiped away with exec.

#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if IS_NONBLOCKING
#  define MAYBE_NONBLOCKING [[clang::nonblocking]]
#else
#  define MAYBE_NONBLOCKING
#endif

int main() MAYBE_NONBLOCKING {
  const pid_t pid = fork();

  if (pid == 0) {
    char *args[] = {"/bin/ls", nullptr};
    execve(args[0], args, nullptr);
    perror("execve failed");
    return 1;
  } else if (pid > 0) {
    int status;
    waitpid(pid, &status, 0);
    usleep(1);
  } else {
    perror("fork failed");
    return 1;
  }

  printf("fork/exec succeeded\n");
  return 0;
}

// CHECK-NOHALT: Intercepted call to {{.*}} `fork` {{.*}}
// CHECK-NOHALT: Intercepted call to {{.*}} `execve` {{.*}}

// usleep checks that rtsan is still enabled in the parent process
// See note in our interceptors file for why we don't look for `wait`
// CHECK-NOHALT: Intercepted call to {{.*}} `usleep` {{.*}}

// CHECK-NOHALT: fork/exec succeeded

// CHECK-HALT: ==ERROR: RealtimeSanitizer: unsafe-library-call
// CHECK-HALT-NEXT: Intercepted call to {{.*}} `fork` {{.*}}

// CHECK-OK: fork/exec succeeded
