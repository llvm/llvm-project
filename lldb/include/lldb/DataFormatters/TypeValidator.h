//===-- TypeValidator.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_TYPEVALIDATOR_H
#define LLDB_DATAFORMATTERS_TYPEVALIDATOR_H

#include "lldb/Symbol/CompilerType.h"

namespace lldb_private {

using CxxTypeValidatorFn = bool(const CompilerType &);

} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_TYPEVALIDATOR_H
