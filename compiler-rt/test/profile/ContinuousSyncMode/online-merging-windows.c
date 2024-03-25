// REQUIRES: target={{.*windows-msvc.*}}
// REQUIRES: lld-available

// Test the online merging mode (%m) along with continuous mode (%c).
//
// Split files & cd into a temporary directory.
// RUN: rm -rf %t.dir && split-file %s %t.dir && cd %t.dir
//
// Create two DLLs and a driver program that uses them.
// RUN: %clang_pgogen foo.c -mllvm -instrprof-atomic-counter-update-all=1 -mllvm -runtime-counter-relocation=true -fuse-ld=lld -Wl,-dll -o %t.dir/foo.dll
// RUN: %clang_pgogen bar.c -mllvm -instrprof-atomic-counter-update-all=1 -mllvm -runtime-counter-relocation=true -fuse-ld=lld -Wl,-dll -o %t.dir/bar.dll
// RUN: %clang_pgogen main.c -o main.exe %t.dir/foo.lib %t.dir/bar.lib -mllvm -instrprof-atomic-counter-update-all=1 -mllvm -runtime-counter-relocation=true -fuse-ld=lld
//
// === Round 1 ===
// Test merging+continuous mode without any file contention.
//
// RUN: env LLVM_PROFILE_FILE="%t.dir/profdir/%m%c.profraw" %run %t.dir/main.exe nospawn
// RUN: llvm-profdata merge -o %t.profdata %t.dir/profdir
// RUN: llvm-profdata show --counts --all-functions %t.profdata | FileCheck %s -check-prefix=ROUND1

// ROUND1-LABEL: Counters:
// ROUND1-DAG:   foo:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-DAG:     Counters: 1
// ROUND1-DAG:     Block counts: [1]
// ROUND1-DAG:   bar:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-DAG:     Counters: 1
// ROUND1-DAG:     Block counts: [1]
// ROUND1-DAG:   main:
// ROUND1-DAG:     Hash: 0x{{.*}}
// ROUND1-LABEL: Instrumentation level: IR
//
// === Round 2 ===
// Test merging+continuous mode with some file contention.
//
// RUN: env LLVM_PROFILE_FILE="%t.dir/profdir/%m%c.profraw" %run %t.dir/main.exe spawn
// RUN: llvm-profdata merge -o %t.profdata %t.dir/profdir
// RUN: llvm-profdata show --counts --all-functions %t.profdata | FileCheck %s -check-prefix=ROUND2

// ROUND2-LABEL: Counters:
// ROUND2-DAG:   foo:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-DAG:     Counters: 1
// ROUND2-DAG:     Block counts: [97]
// ROUND2-DAG:   bar:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-DAG:     Counters: 1
// ROUND2-DAG:     Block counts: [97]
// ROUND2-DAG:   main:
// ROUND2-DAG:     Hash: 0x{{.*}}
// ROUND2-LABEL: Instrumentation level: IR

//--- foo.c
__declspec(dllexport) void foo(void) {}

//--- bar.c
__declspec(dllexport) void bar(void) {}

//--- main.c
#include <stdio.h>
#include <string.h>
#include <windows.h>


const int num_child_procs_to_spawn = 32;

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern char *__llvm_profile_get_filename(void);

__declspec(dllimport) void foo(void);
__declspec(dllimport) void bar(void);

// Change to "#define" for debug output.
#undef DEBUG_TEST

#ifdef DEBUG_TEST
#  define DEBUG(...) fprintf(stderr, __VA_ARGS__);
#else
#  define DEBUG(...)
#endif

int main(int argc, char *const argv[]) {
  if (argc < 2) {
    DEBUG("Requires at least one argument.\n");
    return 1;
  }
  if (strcmp(argv[1], "nospawn") == 0) {
    DEBUG(
        "Hello from child (pid = %lu, cont-mode-enabled = %d, profile = %s).\n",
        GetCurrentProcessId(), __llvm_profile_is_continuous_mode_enabled(),
        __llvm_profile_get_filename());

    foo();
    bar();
    return 0;
  } else if (strcmp(argv[1], "spawn") == 0) {
    // This is the start of Round 2.
    // Expect Counts[dsoX] = 1, as this was the state at the end of Round 1.
    int I;
    HANDLE child_pids[num_child_procs_to_spawn];
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      foo(); // Counts[dsoX] += 2 * num_child_procs_to_spawn
      bar();

      DEBUG("Spawning child with argv = {%s, %s, NULL} and envp = {%s, NULL}\n",
            child_argv[0], child_argv[1], child_envp[0]);

      // Start the child process.
      STARTUPINFO si;
      ZeroMemory(&si, sizeof(si));
      PROCESS_INFORMATION pi;
      ZeroMemory(&pi, sizeof(pi));
      if (!CreateProcess(NULL,               // No module name (use command line)
                         "main.exe nospawn", // Command line
                         NULL,               // Process handle not inheritable
                         NULL,               // Thread handle not inheritable
                         FALSE,              // Set handle inheritance to FALSE
                         0,                  // No creation flags
                         NULL,               // Use parent's environment block
                         NULL,               // Use parent's starting directory
                         &si,                // Pointer to STARTUPINFO structure
                         &pi)                // Pointer to PROCESS_INFORMATION structure
      ) {
        fprintf(stderr, "Child %d could not be spawned: %lu\n", I,
                GetLastError());
        return 1;
      }
      child_pids[I] = pi.hProcess;

      DEBUG("Spawned child %d (pid = %zu).\n", I, pi.dwProcessId);
    }
    for (I = 0; I < num_child_procs_to_spawn; ++I) {
      foo(); // Counts[dsoX] += num_child_procs_to_spawn
      bar();

      DWORD exit_code;
      WaitForSingleObject(child_pids[I], INFINITE);
      if (!GetExitCodeProcess(child_pids[I], &exit_code)) {
        fprintf(stderr, "Failed to get exit code of child %d.\n", I);
        return 1;
      }
      if (exit_code != 0) {
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