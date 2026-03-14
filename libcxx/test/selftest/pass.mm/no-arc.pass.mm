//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: objective-c++

// Make sure ARC is not enabled by default in these tests.

#if __has_feature(objc_arc)
#   error "arc should not be enabled by default"
#endif

int main(int, char**) {
    return 0;
}
