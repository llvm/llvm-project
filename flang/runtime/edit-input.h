//===-- runtime/edit-input.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_EDIT_INPUT_H_
#define FORTRAN_RUNTIME_EDIT_INPUT_H_

#include "format.h"
#include "io-stmt.h"
#include "flang/Decimal/decimal.h"

namespace Fortran::runtime::io {

bool EditIntegerInput(IoStatementState &, const DataEdit &, void *, int kind);

template <int KIND>
bool EditRealInput(IoStatementState &, const DataEdit &, void *);

bool EditLogicalInput(IoStatementState &, const DataEdit &, bool &);
bool EditDefaultCharacterInput(
    IoStatementState &, const DataEdit &, char *, std::size_t);

extern template bool EditRealInput<2>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<3>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<4>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<8>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<10>(
    IoStatementState &, const DataEdit &, void *);
// TODO: double/double
extern template bool EditRealInput<16>(
    IoStatementState &, const DataEdit &, void *);
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_EDIT_INPUT_H_
