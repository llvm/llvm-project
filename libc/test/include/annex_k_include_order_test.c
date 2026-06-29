//===-- Include order test for Annex K macros -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#define __STDC_WANT_LIB_EXT1__ 1
#include <string.h>

_Static_assert(sizeof(strnlen_s("", 0)) == sizeof(size_t),
               "strnlen_s should be declared when Annex K is requested");

int main(void) { return 0; }
