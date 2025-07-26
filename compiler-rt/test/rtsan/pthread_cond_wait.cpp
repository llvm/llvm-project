// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

// Intent: Ensures that pthread_cond_signal does not segfault under rtsan
// See issue #146120

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#include <iostream>

int main() {
  std::cout << "Entry to main!" << std::endl;
  std::mutex mut;
  std::condition_variable cv;
  bool go{false};

  const auto fut = std::async(std::launch::async, [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
      std::unique_lock<std::mutex> lock(mut);
      go = true;
    }
    cv.notify_one();
  });

  std::unique_lock<std::mutex> lock(mut);
  // normal wait is fine
  // cv.wait(lock, [&] { return go; });
  // but timed wait could segfault

  // NOTE: If this test segfaults on a test runner, please comment
  //       out this line and submit the patch.
  //       I will follow up with a fix of the underlying problem,
  //       but first I need to understand if it fails a test runner
  cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return go; });

  std::cout << "Exit from main!" << std::endl;
}

// CHECK: Entry to main!
// CHECK-NEXT: Exit from main!
