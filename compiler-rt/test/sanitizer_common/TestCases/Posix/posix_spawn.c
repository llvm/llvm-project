// RUN: %clang %s -o %t && %run %t 2>&1 | FileCheck %s
//
// Older versions of Android do not have certain posix_spawn* functions.
// UNSUPPORTED: android

// Simulators expect certain envars to be set, but this test overwrites
// env when spawning the child process.
// XFAIL: iossim

#include <assert.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;

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
  char *env[] = {
      "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B",
      "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", "A=B", NULL,
  };

  // When this test runs with a runtime (e.g. ASAN), the spawned process needs
  // to use the same runtime search path as the parent. Otherwise, it might
  // try to load a runtime that doesn't work and crash before hitting main(),
  // failing the test. We technically should forward the variable for the
  // current platform, but some platforms have multiple such variables and
  // it's quite difficult to plumb this through the lit config.
  for (char **e = environ; *e; e++)
    if (strncmp(*e, "DYLD_LIBRARY_PATH=", sizeof("DYLD_LIBRARY_PATH=") - 1) ==
        0)
      env[0] = *e;

  pid_t pid;
  int s = posix_spawn(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);

  s = posix_spawnp(&pid, argv[0], &file_actions, &attr, args, env);
  assert(!s);

  waitpid(pid, &s, WUNTRACED | WCONTINUED);
  return 0;
}
