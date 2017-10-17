//===- Threads.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Threads.h"
#include <thread>
#include <vector>

static std::vector<std::thread> Threads;

bool lld::ThreadsEnabled = true;

// Runs a given function in a new thread.
void lld::runBackground(std::function<void()> Fn) {
  Threads.emplace_back(Fn);
}

// Wait for all threads spawned for runBackground() to finish.
//
// You need to call this function from the main thread before exiting
// because it is not defined what will happen to non-main threads when
// the main thread exits.
void lld::waitForBackgroundThreads() {
  for (std::thread &T : Threads)
    if (T.joinable())
      T.join();
}
