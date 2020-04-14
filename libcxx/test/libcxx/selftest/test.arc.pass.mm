// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-fobjc-arc
// ADDITIONAL_COMPILE_FLAGS: -fobjc-arc

#if __has_feature(objc_arc) == 0
#error "arc should be enabled"
#endif

int main(int, char**) { return 0; }
