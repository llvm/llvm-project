//===-- Unittest for issignaling[f] macro ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsSignalingTest.h"
#include "include/llvm-libc-macros/math-function-macros.h"

// TODO: enable the test unconditionally when issignaling macro is fixed for
//       older compiler
#ifdef issignaling
LIST_ISSIGNALING_TESTS(float, issignaling)
#else
TEST(LlvmLibcIsSignalingTest, Skip) {}
#endif
