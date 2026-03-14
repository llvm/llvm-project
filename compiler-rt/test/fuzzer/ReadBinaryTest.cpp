// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. Tests that fuzzer can read a file containing
// carriage returns.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
  std::string InputStr(reinterpret_cast<const char*>(Data), Size);
  std::string MagicStr("Hello\r\nWorld\r\n");
  if (InputStr == MagicStr) {
    std::cout << "BINGO!";
  }
  return 0;
}
