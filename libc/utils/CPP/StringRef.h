//===----------------- A standalone StringRef type  -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArrayRef.h"

namespace __llvm_libc {
namespace cpp {

class StringRef : public ArrayRef<char> {
  // More methods like those in llvm::StringRef can be added as needed.
};

} // namespace cpp
} // namespace __llvm_libc
