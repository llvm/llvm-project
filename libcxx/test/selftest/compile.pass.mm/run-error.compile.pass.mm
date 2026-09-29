//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: objective-c++

// Make sure the test passes even if there's a runtime error, i.e. it isn't run.

@interface I;
@end

int main(int, char**) {
    return 1;
}
