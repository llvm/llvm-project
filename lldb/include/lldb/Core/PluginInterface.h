//===-- PluginInterface.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PluginInterface_h_
#define liblldb_PluginInterface_h_

#include "lldb/lldb-private.h"

namespace lldb_private {

class PluginInterface {
public:
  virtual ~PluginInterface() {}

  virtual ConstString GetPluginName() = 0;

  virtual uint32_t GetPluginVersion() = 0;
};

} // namespace lldb_private

#endif // liblldb_PluginInterface_h_
