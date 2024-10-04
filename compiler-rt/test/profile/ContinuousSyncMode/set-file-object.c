// REQUIRES: target={{.*(darwin|linux|solaris|aix).*}}

// Test using __llvm_profile_set_file_object in continuous mode (%c).
// Create & cd into a temporary directory.
// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir

// The -mllvm -runtime-counter-relocation=true flag has effect only on linux.
// RUN: %clang -fprofile-instr-generate -fcoverage-mapping -mllvm -instrprof-atomic-counter-update-all=1 -mllvm -runtime-counter-relocation=true -o main.exe %s

// Test continuous mode with __llvm_profile_set_file_object with mergin disabled.
// RUN: env LLVM_PROFILE_FILE="%t.dir/profdir/%c%mprofraw.old" %run  %t.dir/main.exe nomerge %t.dir/profdir/profraw.new 2>&1 | FileCheck %s -check-prefix=WARN
// WARN: LLVM Profile Warning: __llvm_profile_set_file_object(fd={{[0-9]+}}) not supported in continuous sync mode when merging is disabled

// Test continuous mode with __llvm_profile_set_file_object with mergin enabled.
// RUN: rm -rf %t.dir/profdir/
// RUN: env LLVM_PROFILE_FILE="%t.dir/profdir/%c%mprofraw.old" %run  %t.dir/main.exe merge %t.dir/profdir/profraw.new 'LLVM_PROFILE_FILE=%t.dir/profdir/%c%m.profraw'
// RUN: llvm-profdata merge -o %t.dir/profdir/profdata %t.dir/profdir/profraw.new
// RUN: llvm-profdata show --counts --all-functions %t.dir/profdir/profdata | FileCheck %s -check-prefix=MERGE
// RUN: llvm-profdata show --counts --all-functions %t.dir/profdir/*profraw.old | FileCheck %s -check-prefix=ZERO

// Test __llvm_profile_set_file_object with mergin enabled and continuous mode disabled.
// RUN: rm -rf %t.dir/profdir/
// RUN: env LLVM_PROFILE_FILE="%t.dir/profdir/%mprofraw.old" %run  %t.dir/main.exe merge %t.dir/profdir/profraw.new 'LLVM_PROFILE_FILE=%t.dir/profdir/%m.profraw'
// RUN: llvm-profdata merge -o %t.dir/profdir/profdata %t.dir/profdir/profraw.new
// RUN: llvm-profdata show --counts --all-functions %t.dir/profdir/profdata | FileCheck %s -check-prefix=MERGE
// RUN: llvm-profdata show --counts --all-functions %t.dir/profdir/*profraw.old | FileCheck %s -check-prefix=ZERO

// MERGE: Counters:
// MERGE:   coverage_test:
// MERGE:     Hash: {{.*}}
// MERGE:     Counters: 1
// MERGE:     Function count: 32
// MERGE:     Block counts: []
// MERGE: Instrumentation level: Front-end

// ZERO: Counters:
// ZERO:   coverage_test:
// ZERO:     Hash: {{.*}}
// ZERO:     Counters: 1
// ZERO:     Function count: 0
// ZERO:     Block counts: []
// ZERO: Instrumentation level: Front-end

#include <spawn.h>
#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <sys/wait.h>

const int num_child_procs_to_spawn = 32;

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern int __llvm_profile_set_file_object(FILE *, int);

int coverage_test() {
  return 0;
}

int main(int argc, char **argv) {
  char *file_name = argv[2];
  FILE *file = NULL;
  if (strcmp(argv[1], "nomerge") == 0) {
    file = fopen(file_name, "a+b");
    __llvm_profile_set_file_object(file, 0);
  }
  else if (strcmp(argv[1], "merge") == 0) {
    // Parent process.
    int I;
    pid_t child_pids[num_child_procs_to_spawn];
    char *const child_argv[] = {argv[0], "set", file_name, NULL};
    char *const child_envp[] = {argv[3], NULL};
    FILE *file = fopen(file_name, "w+");
    fclose(file);
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      int ret =
          posix_spawn(&child_pids[I], argv[0], NULL, NULL, child_argv, child_envp);
      if (ret != 0) {
        fprintf(stderr, "Child %d could not be spawned: ret = %d, msg = %s\n",
                I, ret, strerror(ret));
        return 1;
      }
    }
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      int status;
      pid_t waited_pid = waitpid(child_pids[I], &status, 0);
      if (waited_pid != child_pids[I]) {
        fprintf(stderr, "Failed to wait on child %d\n", I);
        return 1;
      }
      if (!WIFEXITED(status)) {
        fprintf(stderr, "Child %d did not terminate normally\n", I);
        return 1;
      }
      int return_status = WEXITSTATUS(status);
      if (return_status != 0) {
        fprintf(stderr, "Child %d exited with non zero status %d\n", I,
                return_status);
        return 1;
      }
    }
  } else if (strcmp(argv[1], "set") == 0) {
    // Child processes.
    file = fopen(file_name, "r+b");
    if (__llvm_profile_set_file_object(file, 1)) {
      fprintf(stderr, "Call to __llvm_profile_set_file_object failed\n");
      return 1;
    }
    // After set file object, counter should be written into new file.
    coverage_test();
  }
  return 0;
}
