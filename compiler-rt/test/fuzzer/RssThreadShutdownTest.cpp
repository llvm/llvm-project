// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

// Registered before the RSS monitor is created, so its exit handler must run
// after the monitor has released its thread and its C++ runtime allocations.
static void *volatile LeakSink;

static unsigned CountThreads() {
  DIR *Tasks = opendir("/proc/self/task");
  if (!Tasks)
    abort();
  unsigned Threads = 0;
  while (dirent *Entry = readdir(Tasks))
    if (Entry->d_name[0] >= '0' && Entry->d_name[0] <= '9')
      ++Threads;
  closedir(Tasks);
  return Threads;
}

static void CheckThreadsAtExit() {
  unsigned Threads = CountThreads();
  // The kernel may briefly retain a joined thread's task directory.
  for (unsigned Attempt = 0; Threads != 1 && Attempt != 1000; ++Attempt) {
    usleep(1000);
    Threads = CountThreads();
  }
  if (Threads != 1) {
    fprintf(stderr, "RSS monitor still running at exit: %u threads\n", Threads);
    abort();
  }
  fprintf(stderr, "RSS monitor stopped before process exit\n");
  if (getenv("LIBFUZZER_TEST_LEAK")) {
    LeakSink = malloc(12345);
    LeakSink = nullptr;
  }
}

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  if (atexit(CheckThreadsAtExit))
    abort();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *, size_t) {
  static bool Forked = false;
  if (getenv("LIBFUZZER_TEST_FORK") && !Forked) {
    Forked = true;
    pid_t Child = fork();
    if (Child < 0)
      abort();
    // Run the inherited exit handlers. libFuzzer reports an exit from the
    // target using -error_exitcode=77, even though the target calls exit(0).
    if (Child == 0)
      exit(0);
    int Status;
    pid_t Waited;
    do {
      Waited = waitpid(Child, &Status, 0);
    } while (Waited < 0 && errno == EINTR);
    if (Waited != Child || !WIFEXITED(Status) || WEXITSTATUS(Status) != 77)
      abort();
  }
  return 0;
}
