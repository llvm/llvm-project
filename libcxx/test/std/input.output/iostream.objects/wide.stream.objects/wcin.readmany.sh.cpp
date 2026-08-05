//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// QEMU does not detect EOF, when reading from stdin
// "echo -n" suppresses any characters after the output and so the test hangs.
// https://gitlab.com/qemu-project/qemu/-/issues/1963
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

// This test hangs on Android devices that lack shell_v2, which was added in
// Android N (API 24).
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && android-device-api={{2[1-3]}}

// UNSUPPORTED: no-wide-characters

// <iostream>

// wistream wcin;

// Read an input that is much larger than any reasonable internal buffer so that the
// buffer has to be refilled (underflow) several times, and check that every value
// is read correctly and that EOF is detected at the end.

// FILE_DEPENDENCIES: ../many-ints.dat

// RUN: %{build} -DSYNC_WITH_STDIO_FALSE
// RUN: %{exec} %t.exe < %{temp}/many-ints.dat

// RUN: %{build}
// RUN: %{exec} %t.exe < %{temp}/many-ints.dat

#include <cassert>
#include <iostream>

int main(int, char**) {
#ifdef SYNC_WITH_STDIO_FALSE
  assert(std::ios_base::sync_with_stdio(false));
#endif

  long sum  = 0;
  int count = 0;
  int value = 0;
  while (std::wcin >> value) {
    sum += value;
    ++count;
  }

  assert(count == 5000);
  assert(sum == 5000L * 5001L / 2L); // sum of 1..5000
  assert(std::wcin.eof());
  assert(!std::wcin.good());

  return 0;
}
