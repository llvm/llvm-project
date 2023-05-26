//===-- flang/runtime/non-tbp-dio.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "non-tbp-dio.h"
#include "type-info.h"

namespace Fortran::runtime::io {

const NonTbpDefinedIo *NonTbpDefinedIoTable::Find(
    const typeInfo::DerivedType &type, common::DefinedIo definedIo) const {
  std::size_t j{items};
  for (const auto *p{item}; j-- > 0; ++p) {
    if (&p->derivedType == &type && p->definedIo == definedIo) {
      return p;
    } else if (p->isDtvArgPolymorphic) {
      for (const typeInfo::DerivedType *t{type.GetParentType()}; t;
           t = t->GetParentType()) {
        if (&p->derivedType == t && p->definedIo == definedIo) {
          return p;
        }
      }
    }
  }
  return nullptr;
}

} // namespace Fortran::runtime::io
