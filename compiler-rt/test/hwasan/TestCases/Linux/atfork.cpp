// RUN: %clang_hwasan -O0 %s -o %t && %run %t 2>&1

// REQUIRES: aarch64-target-arch || x86_64-target-arch
// REQUIRES: pointer-tagging

#include <assert.h>
#include <pthread.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

void *volatile sink;

int main(int argc, char **argv) {
  pthread_atfork(nullptr, nullptr, []() {
    alarm(5);
    sink = malloc(10);
  });
  int pid = fork();
  if (pid) {
    int wstatus;
    do {
      waitpid(pid, &wstatus, 0);
    } while (!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus));
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus)) {
      fprintf(stderr, "abnormal exit\n");
      return 1;
    }
  }
  return 0;
}
