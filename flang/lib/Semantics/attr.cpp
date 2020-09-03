//===-- lib/Semantics/attr.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/attr.h"
#include "flang/Common/idioms.h"
#include "llvm/Support/raw_ostream.h"
#include <stddef.h>

namespace Fortran::semantics {

void Attrs::CheckValid(const Attrs &allowed) const {
  if (!allowed.HasAll(*this)) {
    common::die("invalid attribute");
  }
}

std::string AttrToString(Attr attr) {
  switch (attr) {
  case Attr::BIND_C:
    return "BIND(C)";
  case Attr::INTENT_IN:
    return "INTENT(IN)";
  case Attr::INTENT_INOUT:
    return "INTENT(INOUT)";
  case Attr::INTENT_OUT:
    return "INTENT(OUT)";
  default:
    return EnumToString(attr);
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, Attr attr) {
  return o << AttrToString(attr);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const Attrs &attrs) {
  std::size_t n{attrs.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Attr attr{static_cast<Attr>(j)};
    if (attrs.test(attr)) {
      if (seen > 0) {
        o << ", ";
      }
      o << attr;
      ++seen;
    }
  }
  return o;
}
} // namespace Fortran::semantics
