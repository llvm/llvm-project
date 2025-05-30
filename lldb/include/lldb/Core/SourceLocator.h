//===-- SourceLocationSpec.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_SOURCELOCATIOR_H
#define LLDB_UTILITY_SOURCELOCATIOR_H

#include "lldb/Core/PluginInterface.h"

namespace lldb_private {
class SourceLocator : public PluginInterface {};
} // namespace lldb_private

#endif // LLDB_UTILITY_SOURCELOCATIOR_H
