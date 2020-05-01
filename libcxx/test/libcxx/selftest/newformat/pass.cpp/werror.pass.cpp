//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// Make sure that even a simple unused variable warning is treated as an
// error in the test suite. This is to make sure the test suite always runs
// with -Werror.

// ADDITIONAL_COMPILE_FLAGS: -Wunused-variable

// TODO: We don't enable -Werror on GCC right now, because too many tests fail.
// UNSUPPORTED: gcc

// TODO(ldionne): We don't enable -Werror in C++03 right now
// UNSUPPORTED: c++98, c++03

int main() {
    int foo;
}
