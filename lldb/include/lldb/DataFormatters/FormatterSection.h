//===-- FormattersSection.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_FORMATTERSECTION_H
#define LLDB_DATAFORMATTERS_FORMATTERSECTION_H

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {
/// Load type summaries embedded in the binary. These are type summaries
/// provided by the authors of the code.
void LoadTypeSummariesForModule(lldb::ModuleSP module_sp);

/// Load data formatters embedded in the binary. These are formatters provided
/// by the authors of the code using LLDB formatter bytecode.
void LoadFormattersForModule(lldb::ModuleSP module_sp);

} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_FORMATTERSECTION_H
