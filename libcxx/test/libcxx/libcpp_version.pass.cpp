//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that the __libcpp_version file matches the value of _LIBCPP_VERSION

#include <__config>

#ifndef _LIBCPP_VERSION
#error _LIBCPP_VERSION must be defined
#endif

static const int libcpp_version =
#include <__libcpp_version>
;

static_assert(_LIBCPP_VERSION == libcpp_version,
              "_LIBCPP_VERSION doesn't match __libcpp_version");

int main(int, char**) {


  return 0;
}
