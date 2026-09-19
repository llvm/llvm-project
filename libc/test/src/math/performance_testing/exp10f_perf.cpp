//===-- Differential test for exp10f --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfTest.h"
#include "src/math/exp10f.h"

#include <fstream>
#include <math.h>

int main(int argc, char **argv) {
  using Perf = LIBC_NAMESPACE::testing::PerfTest<float, float>;
  using FuncPtr = Perf::UnaryFuncPtr;
  auto libc_func = static_cast<FuncPtr>(&LIBC_NAMESPACE::exp10f);
  auto system_func = static_cast<FuncPtr>(&::exp10f);

  constexpr size_t SAMPLE_COUNT = 1'000'001;
  constexpr size_t ROUND_COUNT = 20;
  constexpr char LOG_FILE[] = "exp10f_perf.log";
  char selected_range = argc > 1 ? argv[1][0] : '\0';

  if (selected_range == '\0')
    Perf::run_perf<false>(libc_func, system_func, 10, "LLVM libc exp10f",
                          "system libc exp10f", LOG_FILE);

  std::ofstream log(LOG_FILE,
                    selected_range == '\0' ? std::ios::app : std::ios::trunc);
  auto run_range = [&](char id, const char *name, uint32_t start,
                       uint32_t stop) {
    if (selected_range != '\0' && selected_range != id)
      return;
    log << "\n " << name << ":\n";
    Perf::run_perf_in_range<false>(
        libc_func, system_func, start, stop, SAMPLE_COUNT, ROUND_COUNT,
        "LLVM libc exp10f", "system libc exp10f", log);
  };

  run_range('1', "Normal outputs, nonnegative inputs", 0x0000'0000U,
            0x421a'209aU);
  run_range('2', "Normal outputs, negative inputs", 0x8000'0000U, 0xc217'b818U);
  run_range('3', "Subnormal outputs", 0xc217'b819U, 0xc234'9e35U);
  run_range('4', "Close to one, nonnegative inputs", 0x0000'0000U,
            0x3b9a'209bU);
  run_range('5', "Close to one, negative inputs", 0x8000'0000U, 0xbb9a'209bU);
  run_range('6', "Overflow inputs", 0x421a'209bU, 0x7f7f'ffffU);
  run_range('7', "Underflow inputs", 0xc234'9e36U, 0xff7f'ffffU);
  return 0;
}
