//===-- strtofloatingpoint comparison test --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "src/__support/str_float_conv_utils.h"

#include <stdlib.h> // For string to float functions

// #include "src/__support/FPUtil/FPBits.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

// The intent of this test is to read in files in the format used in this test
// dataset: https://github.com/nigeltao/parse-number-fxx-test-data
// The format is as follows:
// Hexadecimal representations of IEEE754 floats in 16 bits, 32 bits, and 64
// bits, then the string that matches to them.

// 3C00 3F800000 3FF0000000000000 1.0

// By default, float_comp_in.txt is used as the test set, but once built this
// file can be run against the larger test set. To do that, clone the repository
// with the dataset, then navigate to the compiled binary of this file (it
// should be in llvm_project/build/bin). Run the following command:
// ./libc_str_to_float_comparison_test <path/to/dataset/repo>/data/*
// It will take a few seconds to run.

static inline uint32_t hexCharToU32(char in) {
  return in > '9' ? in + 10 - 'A' : in - '0';
}

// Fast because it assumes inStr points to exactly 8 uppercase hex chars
static inline uint32_t fastHexToU32(const char *inStr) {
  uint32_t result = 0;
  result = (hexCharToU32(inStr[0]) << 28) + (hexCharToU32(inStr[1]) << 24) +
           (hexCharToU32(inStr[2]) << 20) + (hexCharToU32(inStr[3]) << 16) +
           (hexCharToU32(inStr[4]) << 12) + (hexCharToU32(inStr[5]) << 8) +
           (hexCharToU32(inStr[6]) << 4) + hexCharToU32(inStr[7]);
  return result;
}

// Fast because it assumes inStr points to exactly 8 uppercase hex chars
static inline uint64_t fastHexToU64(const char *inStr) {
  uint64_t result = 0;
  result = (static_cast<uint64_t>(fastHexToU32(inStr)) << 32) +
           fastHexToU32(inStr + 8);
  return result;
}

int checkFile(char *inputFileName, int *totalFails, int *totalBitDiffs,
              int *detailedBitDiffs, int *total) {
  int32_t curFails = 0;    // Only counts actual failures, not bitdiffs.
  int32_t curBitDiffs = 0; // A bitdiff is when the expected result and actual
                           // result are off by +/- 1 bit.
  std::string line;
  std::string num;

  std::ifstream fileStream(inputFileName, std::ifstream::in);

  if (!fileStream.is_open()) {
    std::cout << "file '" << inputFileName << "' failed to open. Exiting.\n";
    return 1;
  }
  while (getline(fileStream, line)) {
    if (line[0] == '#') {
      continue;
    }
    *total = *total + 1;
    uint32_t expectedFloatRaw;
    uint64_t expectedDoubleRaw;

    expectedFloatRaw = fastHexToU32(line.c_str() + 5);
    expectedDoubleRaw = fastHexToU64(line.c_str() + 14);
    num = line.substr(31);

    float floatResult = strtof(num.c_str(), nullptr);

    double doubleResult = strtod(num.c_str(), nullptr);

    uint32_t floatRaw = *(uint32_t *)(&floatResult);

    uint64_t doubleRaw = *(uint64_t *)(&doubleResult);

    if (!(expectedFloatRaw == floatRaw)) {
      if (expectedFloatRaw == floatRaw + 1 ||
          expectedFloatRaw == floatRaw - 1) {
        curBitDiffs++;
        if (expectedFloatRaw == floatRaw + 1) {
          detailedBitDiffs[0] = detailedBitDiffs[0] + 1; // float low
        } else {
          detailedBitDiffs[1] = detailedBitDiffs[1] + 1; // float high
        }
      } else {
        curFails++;
      }
      if (curFails + curBitDiffs < 10) {
        std::cout << "Float fail for '" << num << "'. Expected " << std::hex
                  << expectedFloatRaw << " but got " << floatRaw << "\n"
                  << std::dec;
      }
    }

    if (!(expectedDoubleRaw == doubleRaw)) {
      if (expectedDoubleRaw == doubleRaw + 1 ||
          expectedDoubleRaw == doubleRaw - 1) {
        curBitDiffs++;
        if (expectedDoubleRaw == doubleRaw + 1) {
          detailedBitDiffs[2] = detailedBitDiffs[2] + 1; // double low
        } else {
          detailedBitDiffs[3] = detailedBitDiffs[3] + 1; // double high
        }
      } else {
        curFails++;
      }
      if (curFails + curBitDiffs < 10) {
        std::cout << "Double fail for '" << num << "'. Expected " << std::hex
                  << expectedDoubleRaw << " but got " << doubleRaw << "\n"
                  << std::dec;
      }
    }
  }

  fileStream.close();

  *totalBitDiffs += curBitDiffs;
  *totalFails += curFails;

  if (curFails > 1 || curBitDiffs > 1) {
    return 2;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  int result = 0;
  int fails = 0;

  // Bitdiffs are cases where the expected result and actual result only differ
  // by +/- the least significant bit. They are tracked separately from larger
  // failures since a bitdiff is most likely the result of a rounding error, and
  // splitting them off makes them easier to track down.
  int bitdiffs = 0;
  int detailedBitDiffs[4] = {0, 0, 0, 0};

  int total = 0;
  for (int i = 1; i < argc; i++) {
    std::cout << "Starting file " << argv[i] << "\n";
    int curResult =
        checkFile(argv[i], &fails, &bitdiffs, detailedBitDiffs, &total);
    if (curResult == 1) {
      result = 1;
      break;
    } else if (curResult == 2) {
      result = 2;
    }
  }
  std::cout << "Results:\n"
            << "Total significant failed conversions: " << fails << "\n"
            << "Total conversions off by +/- 1 bit: " << bitdiffs << "\n"
            << "\t" << detailedBitDiffs[0] << "\tfloat low\n"
            << "\t" << detailedBitDiffs[1] << "\tfloat high\n"
            << "\t" << detailedBitDiffs[2] << "\tdouble low\n"
            << "\t" << detailedBitDiffs[3] << "\tdouble high\n"
            << "Total lines: " << total << "\n";
  return result;
}
