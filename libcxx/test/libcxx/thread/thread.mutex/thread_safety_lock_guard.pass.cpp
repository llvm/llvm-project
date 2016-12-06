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

// MODULES_DEFINES: _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS
#define _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

#include <mutex>

std::mutex m;
int foo __attribute__((guarded_by(m)));

int main() {
  std::lock_guard<std::mutex> lock(m);
  foo++;
}
