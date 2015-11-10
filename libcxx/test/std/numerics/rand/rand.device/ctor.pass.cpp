//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <random>

// class random_device;

// explicit random_device(const string& token = implementation-defined);

// For the following ctors, the standard states: "The semantics and default
// value of the token parameter are implementation-defined". Implementations
// therefore aren't required to accept any string, but the default shouldn't
// throw.

#include <random>
#include <cassert>
#include <unistd.h>

bool is_valid_random_device(const std::string &token) {
#if defined(_LIBCPP_USING_DEV_RANDOM)
  // Not an exhaustive list: they're the only tokens that are tested below.
  return token == "/dev/urandom" || token == "/dev/random";
#else
  return token == "/dev/urandom";
#endif
}

void check_random_device_valid(const std::string &token) {
  std::random_device r(token);
}

void check_random_device_invalid(const std::string &token) {
  try {
    std::random_device r(token);
    assert(false);
  } catch (const std::system_error &e) {
  }
}

int main() {
  { std::random_device r; }

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

  {
    std::string token = "wrong file";
    if (is_valid_random_device(token))
      check_random_device_valid(token);
    else
      check_random_device_invalid(token);
  }

  {
    std::string token = "/dev/urandom";
    if (is_valid_random_device(token))
      check_random_device_valid(token);
    else
      check_random_device_invalid(token);
  }

  {
    std::string token = "/dev/random";
    if (is_valid_random_device(token))
      check_random_device_valid(token);
    else
      check_random_device_invalid(token);
  }
}
