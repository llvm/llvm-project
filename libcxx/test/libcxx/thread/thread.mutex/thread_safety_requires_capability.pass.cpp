//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: thread-safety

// <mutex>

#define _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

#include <mutex>

std::mutex m;
int foo __attribute__((guarded_by(m)));

void increment() __attribute__((requires_capability(m))) {
  foo++;
}

int main() {
  m.lock();
  increment();
  m.unlock();
}
