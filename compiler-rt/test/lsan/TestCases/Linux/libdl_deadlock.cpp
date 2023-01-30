// Regression test for a deadlock in leak detection,
// where lsan would call dl_iterate_phdr while holding the allocator lock.
// RUN: %clangxx_lsan %s -o %t && %run %t

// Fixme: remove once test passes with hwasan
// UNSUPPORTED: hwasan

#include <link.h>
#include <mutex>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

std::mutex in, out;

int Callback(struct dl_phdr_info *info, size_t size, void *data) {
  for (int step = 0; step < 50; ++step) {
    void *p[1000];
    for (int i = 0; i < 1000; ++i)
      p[i] = malloc(10 * i);

    if (step == 0)
      in.unlock();

    for (int i = 0; i < 1000; ++i)
      free(p[i]);
  }
  out.unlock();
  return 1; // just once
}

void Watchdog() {
  // This is just a fail-safe to turn a deadlock (in case the bug reappears)
  // into a (slow) test failure.
  usleep(40000000);
  if (!out.try_lock()) {
    write(2, "DEADLOCK\n", 9);
    exit(1);
  }
}

int main() {
  in.lock();
  out.lock();

  std::thread t([] { dl_iterate_phdr(Callback, nullptr); });
  t.detach();

  std::thread w(Watchdog);
  w.detach();

  // Wait for the malloc thread to preheat, then start leak detection (on exit)
  in.lock();
  return 0;
}
