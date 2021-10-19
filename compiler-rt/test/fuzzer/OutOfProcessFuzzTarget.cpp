// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is a fuzz target for running out-of-process fuzzing for a
// binary specified via environment variable LIBFUZZER_OOP_TARGET.
// libFuzzer is not designed for out-of-process fuzzing and so this
// ad-hoc rig lacks many of the in-process libFuzzer features, and is slow,
// but it does provide the basic functionality, which is to run the target
// many times in parallel, feeding in the mutants, and expanding the corpus.
// Use this only for very slow targets (slower than ~ 10 exec/s)
// that you can't convert to conventional libFuzzer fuzz targets.
//
// The target binary (which could be a shell script, or anything),
// consumes one file as an input and produces the file with coverage counters
// as the output (output path is passed via SANCOV_OUT).
// One way to produce a valid binary target is to build it with
// -fsanitize-coverage=inline-8bit-counters and link it with SanCovDump.cpp,
// found in the same directory.
//
// Example usage:
/*
 clang -fsanitize=fuzzer OutOfProcessFuzzTarget.cpp -o oop-fuzz &&
 clang -c -fsanitize-coverage=inline-8bit-counters SimpleTest.cpp &&
 clang -c ../../lib/fuzzer/standalone/StandaloneFuzzTargetMain.c &&
 clang -c SanCovDump.cpp &&
 clang++ SanCovDump.o SimpleTest.o  StandaloneFuzzTargetMain.o -o oop-target &&
 rm -rf CORPUS && mkdir CORPUS && echo > CORPUS/seed &&
 LIBFUZZER_OOP_TARGET="./oop-target > /dev/null 2>&1 " ./oop-fuzz CORPUS -jobs=42

*/
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

// An arbitrary large number.
// If your target is so large that it has more than this number of coverage
// edges, you may want to increase this number to match your binary,
// otherwise part of the coverage will be lost.
// For small targets there is no reason to reduce this number.
static const size_t kCountersSize = 1 << 20;

__attribute__((section(
    "__libfuzzer_extra_counters"))) static uint8_t Counters[kCountersSize];

static std::string *Run, *IN, *COV;

void TearDown() {
  unlink(COV->c_str());
  unlink(IN->c_str());
}

bool Initialize() {
  IN = new std::string("lf-oop-in-" + std::to_string(getpid()));
  COV = new std::string("lf-oop-cov-" + std::to_string(getpid()));
  const char *TargetEnv = getenv("LIBFUZZER_OOP_TARGET");
  if (!TargetEnv) {
    fprintf(stderr, "Please define LIBFUZZER_OOP_TARGET\n");
    exit(1);
  }
  Run = new std::string("SANCOV_OUT=" + *COV + " " + TargetEnv + " " + *IN);
  fprintf(stderr, "libFuzzer: OOP command: %s\n", Run->c_str());
  atexit(TearDown);
  return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static bool Inited = Initialize();
  if (size == 0)
    return 0;
  if (FILE *f = fopen(IN->c_str(), "w")) {
    fwrite(data, 1, size, f);
    fclose(f);
  }
  system(Run->c_str());
  if (FILE *f = fopen(COV->c_str(), "r")) {
    fread(Counters, 1, kCountersSize, f);
    fclose(f);
  }
  return 0;
}
