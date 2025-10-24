//===-- Differential test for logbf----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfTest.h"
#include "src/math/logbf.h"

#include <math.h>

int main() {
  SINGLE_INPUT_SINGLE_OUTPUT_PERF(float, LIBC_NAMESPACE::logbf, ::logbf,
                                  "logbf_perf.log")
  return 0;
}
