//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that std::random_device is not available in namespace std:: when
// libc++ is built without support for random device.

// REQUIRES: no-random-device

#include <random>

int main(int, char**) {
  std::random_device d; // expected-error {{no type named 'random_device' in namespace 'std'}}
  return 0;
}
