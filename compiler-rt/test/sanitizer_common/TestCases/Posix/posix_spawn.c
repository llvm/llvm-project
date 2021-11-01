// RUN: %clang %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <spawn.h>
#include <stdio.h>
#include <wait.h>

int main(int argc, char **argv) {
  if (argc > 1) {
    // CHECK: SPAWNED
    // CHECK: SPAWNED
    printf("SPAWNED\n");
    return 0;
  }

  int s;

  posix_spawnattr_t attr;
  s = posix_spawnattr_init(&attr);
  assert(!s);

  posix_spawn_file_actions_t file_actions;
  s = posix_spawn_file_actions_init(&file_actions);
  assert(!s);

  char *const args[] = {argv[0], "2", NULL};
  char *const env[] = {"A=B", NULL};

  pid_t pid;
  s = posix_spawn(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);

  s = posix_spawnp(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);
  return 0;
}
