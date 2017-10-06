//===- Threads.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Threads.h"
#include <thread>

static std::vector<std::thread> Threads;

// Runs a given function in a new thread.
void lld::elf::runBackground(std::function<void()> Fn) {
  Threads.emplace_back(Fn);
}

// Wait for all threads spawned for runBackground() to finish.
//
// You need to call this function from the main thread before exiting
// because it is not defined what will happen to non-main threads when
// the main thread exits.
void lld::elf::waitForBackgroundThreads() {
  for (std::thread &T : Threads)
    if (T.joinable())
      T.join();
}
