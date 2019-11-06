//===-- STLUtils.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_STLUtils_h_
#define liblldb_STLUtils_h_

#include <string.h>

#include <map>
#include <ostream>
#include <vector>


// C string less than compare function object
struct CStringCompareFunctionObject {
  bool operator()(const char *s1, const char *s2) const {
    return strcmp(s1, s2) < 0;
  }
};

#endif // liblldb_STLUtils_h_
