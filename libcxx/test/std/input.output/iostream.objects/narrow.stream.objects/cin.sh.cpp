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

// This test pipes stdin via adb_run.py and hangs on Android M (API 23) because
// adbd before Android N (API 24) lacks shell_v2 stdin EOF propagation. Resolve
// by redirecting stdin from a pushed file in adb_run.py on pre-N devices, or
// remove once API 23 testing is dropped.
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && android-device-api=23

// <iostream>

// istream cin;

// RUN: %{build}
// RUN: echo -n 1234 > %t.input
// RUN: %{exec} %t.exe < %t.input

#include <iostream>
#include <cassert>

int main(int, char**) {
    int i;
    std::cin >> i;
    assert(i == 1234);
    return 0;
}
