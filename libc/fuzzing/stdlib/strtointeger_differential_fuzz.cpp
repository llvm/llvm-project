//===-- strtointeger_differential_fuzz.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc atof implementation.
///
//===----------------------------------------------------------------------===//
#include "src/stdlib/atoi.h"
#include "src/stdlib/atol.h"
#include "src/stdlib/atoll.h"
#include "src/stdlib/strtol.h"
#include "src/stdlib/strtoll.h"
#include "src/stdlib/strtoul.h"
#include "src/stdlib/strtoull.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "fuzzing/stdlib/StringParserOutputDiff.h"

// This list contains (almost) all character that can possibly be accepted by a
// string to integer conversion. Those are: space, tab, +/- signs, any digit,
// and any letter. Technically there are some space characters accepted by
// isspace that aren't in this list, but since space characters are just skipped
// over anyways I'm not really worried.
[[maybe_unused]] constexpr char VALID_CHARS[] = {
    ' ', '\t', '-', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'A',  'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G',
    'h', 'H',  'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N',
    'o', 'O',  'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U',
    'v', 'V',  'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z'};

// This takes the randomized bytes in data and interprets the first byte as the
// base for the string to integer conversion and the rest of them as a string to
// be passed to the string to integer conversion.
// If the CLEANER_INPUT flag is set, the string is modified so that it's only
// made of characters that the string to integer functions could accept. This is
// because every other character is effectively identical, and will be treated
// as the end of the integer. For the fully randomized string this gives a
// greater than 50% chance for each character to end the string, making the odds
// of getting long numbers very low.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2) // Needs at least one byte for the base and one byte for the
                // string.
    return 0;

  uint8_t *container = new uint8_t[size + 1];
  if (!container)
    __builtin_trap();
  size_t i;

  for (i = 0; i < size; ++i) {
#ifdef LIBC_COPT_FUZZ_ATOI_CLEANER_INPUT
    container[i] = VALID_CHARS[data[i] % sizeof(VALID_CHARS)];
#else
    container[i] = data[i];
#endif
  }
  container[size] = '\0'; // Add null terminator to container.
  // the first character is interpreted as the base, so it should be fully
  // random even when the input is cleaned.
  container[0] = data[0];

  StringParserOutputDiff<int>(&LIBC_NAMESPACE::atoi, &::atoi, container, size);
  StringParserOutputDiff<long>(&LIBC_NAMESPACE::atol, &::atol, container, size);
  StringParserOutputDiff<long long>(&LIBC_NAMESPACE::atoll, &::atoll, container,
                                    size);

  StringToNumberOutputDiff<long>(&LIBC_NAMESPACE::strtol, &::strtol, container,
                                 size);
  StringToNumberOutputDiff<long long>(&LIBC_NAMESPACE::strtoll, &::strtoll,
                                      container, size);

  StringToNumberOutputDiff<unsigned long>(&LIBC_NAMESPACE::strtoul, &::strtoul,
                                          container, size);
  StringToNumberOutputDiff<unsigned long long>(&LIBC_NAMESPACE::strtoull,
                                               &::strtoull, container, size);

  delete[] container;
  return 0;
}
