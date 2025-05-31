//===-- lib/Support/enum-class.cpp -------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/enum-class.h"
#include "flang/Common/optional.h"
#include <functional>

namespace Fortran::common::EnumClass {

optional<std::size_t> FindIndex(
    std::function<bool(const std::string_view)> pred, size_t size,
    const std::string_view *names) {
  for (size_t i = 0; i < size; ++i) {
    if (pred(names[i])) {
      return i;
    }
  }
  return nullopt;
}

} // namespace Fortran::common::EnumClass
