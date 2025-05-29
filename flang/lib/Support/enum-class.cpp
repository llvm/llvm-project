//===-- lib/Support/enum-class.cpp -------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/enum-class.h"
#include <functional>
#include <optional>
namespace Fortran::common {

std::optional<int> FindEnumIndex(
    std::function<bool(const std::string_view)> pred, int size,
    const std::string_view *names) {
  for (int i = 0; i < size; ++i) {
    if (pred(names[i])) {
      return i;
    }
  }
  return std::nullopt;
}

} // namespace Fortran::common