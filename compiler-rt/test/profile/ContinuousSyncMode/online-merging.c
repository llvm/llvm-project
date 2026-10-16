// REQUIRES: continuous-mode
// UNSUPPORTED: powerpc-{{.*}}

// Test the online merging mode (%m) along with continuous mode (%c).
//
// Create & cd into a temporary directory.
// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir
//
// Create two DSOs and a driver program that uses them.
// RUN: echo "void dso1(void) {}" > dso1.c
// RUN: echo "void dso2(void) {}" > dso2.c
// RUN: %clang_pgogen=%t.dir/profdir -fprofile-continuous -fprofile-update=atomic %shared_lib_flag -o %t.dir/dso1.dylib dso1.c
// RUN: %clang_pgogen=%t.dir/profdir -fprofile-continuous -fprofile-update=atomic %shared_lib_flag -o %t.dir/dso2.dylib dso2.c
// RUN: %clang_pgogen=%t.dir/profdir -fprofile-continuous -fprofile-update=atomic -o main.exe %s %t.dir/dso1.dylib %t.dir/dso2.dylib
//
// === Round 1 ===
// Test merging+continuous mode without any file contention.
//
// RUN: %run %t.dir/main.exe nospawn
// RUN: llvm-profdata merge -o %t.profdata %t.dir/profdir
// RUN: llvm-profdata show --counts --all-functions %t.profdata | FileCheck %s -check-prefix=ROUND1

// ROUND1-LABEL: Counters:
// ROUND1-DAG:   dso1:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-DAG:     Counters: 1
// ROUND1-DAG:     Block counts: [1]
// ROUND1-DAG:   dso2:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-DAG:     Counters: 1
// ROUND1-DAG:     Block counts: [1]
// ROUND1-DAG:   main:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-LABEL: Instrumentation level: IR
// ROUND1-NEXT: Functions shown: 3
// ROUND1-NEXT: Total functions: 3
// ROUND1-NEXT: Maximum function count: 1
// ROUND1-NEXT: Maximum internal block count: 1
//
// === Round 2 ===
// Test merging+continuous mode with some file contention.
//
// RUN: %run %t.dir/main.exe spawn 'LLVM_PROFILE_FILE=%t.dir/profdir/%m%c.profraw'
// RUN: llvm-profdata merge -o %t.profdata %t.dir/profdir
// RUN: llvm-profdata show --counts --all-functions %t.profdata | FileCheck %s -check-prefix=ROUND2

// ROUND2-LABEL: Counters:
// ROUND2-DAG:   dso1:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-DAG:     Counters: 1
// ROUND2-DAG:     Block counts: [97]
// ROUND2-DAG:   dso2:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-DAG:     Counters: 1
// ROUND2-DAG:     Block counts: [97]
// ROUND2-DAG:   main:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-LABEL: Instrumentation level: IR
// ROUND2-NEXT: Functions shown: 3
// ROUND2-NEXT: Total functions: 3
// ROUND2-NEXT: Maximum function count: 97
// ROUND2-NEXT: Maximum internal block count: 33

#include <spawn.h>
#include <sys/wait.h>
#include <sys/errno.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

const int num_child_procs_to_spawn = 32;

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern char *__llvm_profile_get_filename(void);

void dso1(void);
void dso2(void);

// Change to "#define" for debug output.
#undef DEBUG_TEST

#ifdef DEBUG_TEST
#define DEBUG(...) fprintf(stderr, __VA_ARGS__);
#else
#define DEBUG(...)
#endif

int main(int argc, char *const argv[]) {
  if (strcmp(argv[1], "nospawn") == 0) {
    DEBUG("Hello from child (pid = %d, cont-mode-enabled = %d, profile = %s).\n",
        getpid(), __llvm_profile_is_continuous_mode_enabled(), __llvm_profile_get_filename());

    dso1();
    dso2();
    return 0;
  } else if (strcmp(argv[1], "spawn") == 0) {
    // This is the start of Round 2.
    // Expect Counts[dsoX] = 1, as this was the state at the end of Round 1.

    int I;
    pid_t child_pids[num_child_procs_to_spawn];
    char *const child_argv[] = {argv[0], "nospawn", NULL};
    char *const child_envp[] = {argv[2], NULL};
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      dso1(); // Counts[dsoX] += 2 * num_child_procs_to_spawn
      dso2();

      DEBUG("Spawning child with argv = {%s, %s, NULL} and envp = {%s, NULL}\n",
          child_argv[0], child_argv[1], child_envp[0]);

      int ret = posix_spawn(&child_pids[I], argv[0], NULL, NULL, child_argv,
          child_envp);
      if (ret != 0) {
        fprintf(stderr, "Child %d could not be spawned: ret = %d, msg = %s\n",
                I, ret, strerror(ret));
        return 1;
      }

      DEBUG("Spawned child %d (pid = %d).\n", I, child_pids[I]);
    }
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      dso1(); // Counts[dsoX] += num_child_procs_to_spawn
      dso2();

      int status;
      waitpid(child_pids[I], &status, 0);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "Child %d did not exit with code 0.\n", I);
        return 1;
      }
    }

    // At the end of Round 2, we have:
    // Counts[dsoX] = 1 + (2 * num_child_procs_to_spawn) + num_child_procs_to_spawn
    //              = 97

    return 0;
  }

  return 1;
}
