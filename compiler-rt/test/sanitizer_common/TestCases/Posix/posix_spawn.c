// RUN: %clang %s -o %t && %run %t 2>&1 | FileCheck %s
//
// Older versions of Android do not have certain posix_spawn* functions.
// UNSUPPORTED: android

#include <assert.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>

int main(int argc, char **argv) {
  if (argc > 1) {
    // CHECK: SPAWNED
    // CHECK: SPAWNED
    printf("SPAWNED\n");
    return 0;
  }

  posix_spawnattr_t attr = {0};
  posix_spawn_file_actions_t file_actions = {0};

  char *const args[] = {
      argv[0], "2", "3", "4", "2", "3", "4", "2", "3", "4",
      "2",     "3", "4", "2", "3", "4", "2", "3", "4", NULL,
  };
  char *const env[] = {
      "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B",
      "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", NULL,
  };

  pid_t pid;
  int s = posix_spawn(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);

  s = posix_spawnp(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);
  return 0;
}
