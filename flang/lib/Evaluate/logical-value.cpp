//===-- lib/Evaluate/logical-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/logical-value.h"

namespace Fortran::evaluate::value {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void LogicalValue::dump() const {
  if (!IsCanonical()) {
    llvm::errs() << "transfer(" << word().ToInt64() << "_8,.false._" << kind()
                 << ")\n";
  } else if (IsTrue()) {
    llvm::errs() << ".true." << '_' << kind() << '\n';
  } else {
    llvm::errs() << ".false." << '_' << kind() << '\n';
  }
}
#endif

} // namespace Fortran::evaluate::value
