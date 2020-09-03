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

template <int binaryPrecision>
bool EditRealInput(IoStatementState &, const DataEdit &, void *);

bool EditLogicalInput(IoStatementState &, const DataEdit &, bool &);
bool EditDefaultCharacterInput(
    IoStatementState &, const DataEdit &, char *, std::size_t);

extern template bool EditRealInput<8>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<11>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<24>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<53>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<64>(
    IoStatementState &, const DataEdit &, void *);
extern template bool EditRealInput<113>(
    IoStatementState &, const DataEdit &, void *);
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_EDIT_INPUT_H_
