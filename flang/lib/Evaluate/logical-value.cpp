//===-- lib/Evaluate/logical-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/logical-value.h"

namespace Fortran::evaluate::value {

void LogicalValue::print(llvm::raw_ostream &os) const {
  if (!IsCanonical()) {
    // PAPAYA: This was modified from formatting.cpp where kind 8 is hardcoded
    os << "transfer(";
    word().print(os);
    os << ",.false._" << kind() << ')';
  } else if (IsTrue()) {
    os << ".true." << '_' << kind();
  } else {
    os << ".false." << '_' << kind();
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void LogicalValue::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}
#endif

} // namespace Fortran::evaluate::value
