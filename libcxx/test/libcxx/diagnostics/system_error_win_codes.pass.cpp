//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: windows

// Validate that system_error on windows accepts Windows' System Error Codes (as
// used by win32 APIs and reported by GetLastError), and that they are properly
// translated to generic conditions.

#include <windows.h>
#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  LIBCPP_ASSERT(std::error_code(ERROR_ACCESS_DENIED, std::system_category()) == std::errc::permission_denied);
  LIBCPP_ASSERT(std::error_code(ERROR_PATH_NOT_FOUND, std::system_category()) == std::errc::no_such_file_or_directory);
  return 0;
}
