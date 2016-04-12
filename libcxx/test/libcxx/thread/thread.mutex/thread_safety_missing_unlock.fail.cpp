//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads
// REQUIRES: thread-safety

// <mutex>

#define _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

#include <mutex>

std::mutex m;

int main() {
  m.lock();
} // expected-error {{mutex 'm' is still held at the end of function}}
