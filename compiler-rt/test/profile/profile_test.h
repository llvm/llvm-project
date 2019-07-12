//===-- profile_test.h.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the profile tests.
//===----------------------------------------------------------------------===//
#ifndef PROFILE_TEST_H
#define PROFILE_TEST_H

#if defined(_MSC_VER)
# define ALIGNED(x) __declspec(align(x))
#else  // _MSC_VER
# define ALIGNED(x) __attribute__((aligned(x)))
#endif

#endif  // PROFILE_TEST_H
