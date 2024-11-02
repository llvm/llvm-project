//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// See https://llvm.org/PR20183
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11}}

// The behavior of std::random_device changed on Apple platforms with
// https://llvm.org/D116045.
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

// UNSUPPORTED: no-random-device

// <random>

// class random_device;

// explicit random_device(const string& token = implementation-defined); // before C++20
// random_device() : random_device(implementation-defined) {}            // C++20
// explicit random_device(const string& token);                          // C++20

// For the following ctors, the standard states: "The semantics and default
// value of the token parameter are implementation-defined". Implementations
// therefore aren't required to accept any string, but the default shouldn't
// throw.

#include <random>
#include <string>
#include <system_error>
#include <cassert>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

void check_random_device_valid(const std::string &token) {
  std::random_device r(token);
}

void check_random_device_invalid(const std::string &token) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    std::random_device r(token);
    LIBCPP_ASSERT(false);
  } catch (const std::system_error&) {
  }
#else
  ((void)token);
#endif
}

int main(int, char**) {
  {
    std::random_device r;
    (void)r;
  }
  // Check the validity of various tokens
  {
#if defined(_LIBCPP_USING_ARC4_RANDOM)
    check_random_device_valid("/dev/urandom");
    check_random_device_valid("/dev/random");
    check_random_device_valid("/dev/null");
    check_random_device_valid("/dev/nonexistent");
    check_random_device_valid("wrong file");
#elif defined(_LIBCPP_USING_DEV_RANDOM)
    check_random_device_valid("/dev/urandom");
    check_random_device_valid("/dev/random");
    check_random_device_valid("/dev/null");
    check_random_device_invalid("/dev/nonexistent");
    check_random_device_invalid("wrong file");
#else
    check_random_device_valid("/dev/urandom");
    check_random_device_invalid("/dev/random");
    check_random_device_invalid("/dev/null");
    check_random_device_invalid("/dev/nonexistent");
    check_random_device_invalid("wrong file");
#endif
  }

#if !defined(_WIN32)
// Test that random_device(const string&) properly handles getting
// a file descriptor with the value '0'. Do this by closing the standard
// streams so that the descriptor '0' is available.
  {
    int ec;
    ec = close(STDIN_FILENO);
    assert(!ec);
    ec = close(STDOUT_FILENO);
    assert(!ec);
    ec = close(STDERR_FILENO);
    assert(!ec);
    std::random_device r;
  }
#endif // !defined(_WIN32)

#if TEST_STD_VER >= 11
  static_assert(test_convertible<std::random_device>(), "");
#endif

  return 0;
}
