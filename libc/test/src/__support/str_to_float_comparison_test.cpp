//===-- strtofloatingpoint comparison test --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fopen.h"
#include "src/stdio/printf.h"
#include "src/stdlib/getenv.h"
#include "src/stdlib/strtod.h"
#include "src/stdlib/strtof.h"
#include "src/string/strdup.h"
#include "src/string/strtok.h"
#include "test/UnitTest/Test.h"
#include <stdint.h>

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

struct ParseResult {
  uint32_t totalFails;
  uint32_t totalBitDiffs;
  uint32_t detailedBitDiffs[4];
  uint32_t total;
};

enum class ParseStatus : uint8_t {
  SUCCESS,
  FILE_ERROR,
  PARSE_ERROR,
};

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

static void parseLine(const char *line, ParseResult &parseResult,
                      int32_t &curFails, int32_t &curBitDiffs) {

  if (line[0] == '#')
    return;

  parseResult.total += 1;
  uint32_t expectedFloatRaw;
  uint64_t expectedDoubleRaw;

  expectedFloatRaw = fastHexToU32(line + 5);
  expectedDoubleRaw = fastHexToU64(line + 14);

  const char *num = line + 31;

  float floatResult = LIBC_NAMESPACE::strtof(num, nullptr);

  double doubleResult = LIBC_NAMESPACE::strtod(num, nullptr);

  uint32_t floatRaw = LIBC_NAMESPACE::cpp::bit_cast<uint32_t>(floatResult);

  uint64_t doubleRaw = LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(doubleResult);

  if (!(expectedFloatRaw == floatRaw)) {
    if (expectedFloatRaw == floatRaw + 1 || expectedFloatRaw == floatRaw - 1) {
      curBitDiffs++;
      if (expectedFloatRaw == floatRaw + 1) {
        parseResult.detailedBitDiffs[0] =
            parseResult.detailedBitDiffs[0] + 1; // float low
      } else {
        parseResult.detailedBitDiffs[1] =
            parseResult.detailedBitDiffs[1] + 1; // float high
      }
    } else {
      curFails++;
    }
    if (curFails + curBitDiffs < 10) {
      LIBC_NAMESPACE::printf("Float fail for '%s'. Expected %x but got %x\n",
                             num, expectedFloatRaw, floatRaw);
    }
  }

  if (!(expectedDoubleRaw == doubleRaw)) {
    if (expectedDoubleRaw == doubleRaw + 1 ||
        expectedDoubleRaw == doubleRaw - 1) {
      curBitDiffs++;
      if (expectedDoubleRaw == doubleRaw + 1) {
        parseResult.detailedBitDiffs[2] =
            parseResult.detailedBitDiffs[2] + 1; // double low
      } else {
        parseResult.detailedBitDiffs[3] =
            parseResult.detailedBitDiffs[3] + 1; // double high
      }
    } else {
      curFails++;
    }
    if (curFails + curBitDiffs < 10) {
      LIBC_NAMESPACE::printf("Double fail for '%s'. Expected %lx but got %lx\n",
                             num, expectedDoubleRaw, doubleRaw);
    }
  }
}

ParseStatus checkBuffer(ParseResult &parseResult) {
  constexpr const char *LINES[] = {
      "3C00 3F800000 3FF0000000000000 1",
      "3D00 3FA00000 3FF4000000000000 1.25",
      "3D9A 3FB33333 3FF6666666666666 1.4",
      "57B7 42F6E979 405EDD2F1A9FBE77 123.456",
      "622A 44454000 4088A80000000000 789",
      "7C00 7F800000 7FF0000000000000 123.456e789"};

  int32_t curFails = 0;    // Only counts actual failures, not bitdiffs.
  int32_t curBitDiffs = 0; // A bitdiff is when the expected result and actual
  // result are off by +/- 1 bit.

  for (uint8_t i = 0; i < sizeof(LINES) / sizeof(LINES[0]); i++) {
    parseLine(LINES[i], parseResult, curFails, curBitDiffs);
  }

  parseResult.totalBitDiffs += curBitDiffs;
  parseResult.totalFails += curFails;

  if (curFails > 1 || curBitDiffs > 1) {
    return ParseStatus::PARSE_ERROR;
  }
  return ParseStatus::SUCCESS;
}

ParseStatus checkFile(char *inputFileName, ParseResult &parseResult) {
  int32_t curFails = 0;    // Only counts actual failures, not bitdiffs.
  int32_t curBitDiffs = 0; // A bitdiff is when the expected result and actual
  // result are off by +/- 1 bit.
  char line[1000];

  auto *fileHandle = LIBC_NAMESPACE::fopen(inputFileName, "r");

  if (!fileHandle) {
    LIBC_NAMESPACE::printf("file '%s' failed to open. Exiting.\n",
                           inputFileName);
    return ParseStatus::FILE_ERROR;
  }

  while (LIBC_NAMESPACE::fgets(line, sizeof(line), fileHandle)) {
    parseLine(line, parseResult, curFails, curBitDiffs);
  }

  LIBC_NAMESPACE::fclose(fileHandle);

  parseResult.totalBitDiffs += curBitDiffs;
  parseResult.totalFails += curFails;

  if (curFails > 1 || curBitDiffs > 1) {
    return ParseStatus::PARSE_ERROR;
  }
  return ParseStatus::SUCCESS;
}

ParseStatus updateStatus(ParseStatus parse_status, ParseStatus cur_status) {
  if (cur_status == ParseStatus::FILE_ERROR) {
    parse_status = ParseStatus::FILE_ERROR;
  } else if (cur_status == ParseStatus::PARSE_ERROR) {
    parse_status = ParseStatus::PARSE_ERROR;
  }
  return parse_status;
}

TEST(LlvmLibcStrToFloatComparisonTest, CheckFloats) {
  ParseStatus parseStatus = ParseStatus::SUCCESS;

  // Bitdiffs are cases where the expected result and actual result only differ
  // by +/- the least significant bit. They are tracked separately from larger
  // failures since a bitdiff is most likely the result of a rounding error, and
  // splitting them off makes them easier to track down.

  ParseResult parseResult = {
      .totalFails = 0,
      .totalBitDiffs = 0,
      .detailedBitDiffs = {0, 0, 0, 0},
      .total = 0,
  };

  char *files = LIBC_NAMESPACE::getenv("FILES");

  if (files == nullptr) {
    ParseStatus cur_status = checkBuffer(parseResult);
    parseStatus = updateStatus(parseStatus, cur_status);
  } else {
    files = LIBC_NAMESPACE::strdup(files);
    for (char *file = LIBC_NAMESPACE::strtok(files, ","); file != nullptr;
         file = LIBC_NAMESPACE::strtok(nullptr, ",")) {
      ParseStatus cur_status = checkFile(file, parseResult);
      parseStatus = updateStatus(parseStatus, cur_status);
    }
  }

  EXPECT_EQ(parseStatus, ParseStatus::SUCCESS);
  EXPECT_EQ(parseResult.totalFails, 0u);
  EXPECT_EQ(parseResult.totalBitDiffs, 0u);
  EXPECT_EQ(parseResult.detailedBitDiffs[0], 0u); // float low
  EXPECT_EQ(parseResult.detailedBitDiffs[1], 0u); // float high
  EXPECT_EQ(parseResult.detailedBitDiffs[2], 0u); // double low
  EXPECT_EQ(parseResult.detailedBitDiffs[3], 0u); // double high
  LIBC_NAMESPACE::printf("Total lines: %d\n", parseResult.total);
}
