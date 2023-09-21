//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Investigate
// UNSUPPORTED: LIBCXX-AIX-FIXME

// TODO: Make it possible to run this test when cross-compiling and running via a SSH executor
//       This is a workaround to silence issues reported in https://github.com/llvm/llvm-project/pull/66842#issuecomment-1728701639
// XFAIL: buildhost=windows && target={{.+}}-linux-{{.+}}

// <iostream>

// wistream wcin;

// XFAIL: no-wide-characters

// RUN: %{build}
// RUN: echo -n 1234 | %{exec} %t.exe

#include <iostream>
#include <cassert>

int main(int, char**) {
    int i;
    std::wcin >> i;
    assert(i == 1234);
    return 0;
}
