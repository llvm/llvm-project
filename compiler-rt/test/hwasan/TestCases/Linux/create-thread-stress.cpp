// Stress test for https://reviews.llvm.org/D101881
// RUN: %clangxx_hwasan -DREUSE=0 %s -pthread -O2 -o %t && %run %t 2>&1
// RUN: %clangxx_hwasan -DREUSE=1 %s -pthread -O2 -o %t_reuse && %run %t_reuse 2>&1

#include <thread>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <stdio.h>

constexpr int kTopThreads = 20;
constexpr int kChildThreads = 30;
constexpr int kChildIterations = REUSE ? 8 : 1;

constexpr int kProcessIterations = 20;

void Thread() {
  for (int i = 0; i < kChildIterations; ++i) {
    std::vector<std::thread> threads;
    for (int i = 0; i < kChildThreads; ++i)
      threads.emplace_back([]() {});
    for (auto &t : threads)
      t.join();
  }
}

void run() {
  std::vector<std::thread> threads;
  for (int i = 0; i < kTopThreads; ++i)
    threads.emplace_back(Thread);
  for (auto &t : threads)
    t.join();
}

int main() {
#if REUSE
  // Test thread reuse with multiple iterations of thread create / join in a single process.
  run();
#else
  // Test new, non-reused thread creation by running a single iteration of create / join in a freshly started process.
  for (int i = 0; i < kProcessIterations; ++i) {
    int pid = fork();
    if (pid) {
      int wstatus;
      do {
        waitpid(pid, &wstatus, 0);
      } while (!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus));
      if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus)) {
        fprintf(stderr, "failed at iteration %d / %d\n", i, kProcessIterations);
        return 1;
      }
    } else {
      run();
      return 0;
    }
  }
#endif
}
