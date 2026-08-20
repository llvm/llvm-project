//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_BUGREPORTER_H
#define LLDB_CORE_BUGREPORTER_H

#include "lldb/Core/Diagnostics.h"
#include "lldb/Core/PluginInterface.h"

#include "llvm/Support/Error.h"

namespace lldb_private {

/// A pluggable destination for a diagnostics bundle.
/// CreateBugReporterInstance() returns the first registered reporter, so
/// downstream registers ahead of the no-op fallback to take over.
class BugReporter : public PluginInterface {
public:
  virtual llvm::Error File(const Diagnostics::Report &report) = 0;
};

} // namespace lldb_private

#endif // LLDB_CORE_BUGREPORTER_H
