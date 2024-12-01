//===-- Implementation of the file path generator for bazel ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

#include <stdlib.h>

#include "src/__support/CPP/string.h"
#include "src/__support/c_string.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

CString libc_make_test_file_path_func(const char *file_name) {
  // This is the path to the folder bazel wants the test outputs written to.
  const char *UNDECLARED_OUTPUTS_PATH = getenv("TEST_UNDECLARED_OUTPUTS_DIR");

  return cpp::string(UNDECLARED_OUTPUTS_PATH) + file_name;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
