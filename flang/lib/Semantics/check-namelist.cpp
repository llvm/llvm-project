//===-- lib/Semantics/check-namelist.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-namelist.h"

namespace Fortran::semantics {

void NamelistChecker::Leave(const parser::NamelistStmt &nmlStmt) {
  for (const auto &x : nmlStmt.v) {
    if (const auto *nml{std::get<parser::Name>(x.t).symbol}) {
      for (const auto &nmlObjName : std::get<std::list<parser::Name>>(x.t)) {
        const auto *nmlObjSymbol{nmlObjName.symbol};
        if (nmlObjSymbol) {
          if (IsAssumedSizeArray(*nmlObjSymbol)) { // C8104
            context_.Say(nmlObjName.source,
                "A namelist group object '%s' must not be assumed-size"_err_en_US,
                nmlObjSymbol->name());
          }
          if (nml->attrs().test(Attr::PUBLIC) &&
              nmlObjSymbol->attrs().test(Attr::PRIVATE)) { // C8105
            context_.Say(nmlObjName.source,
                "A PRIVATE namelist group object '%s' must not be in a "
                "PUBLIC namelist"_err_en_US,
                nmlObjSymbol->name());
          }
        }
      }
    }
  }
}

} // namespace Fortran::semantics
