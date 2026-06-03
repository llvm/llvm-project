//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H
#define LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <string>

namespace lldb_private {

struct AcceleratorActions {
  AcceleratorActions() = default;
  AcceleratorActions(llvm::StringRef plugin_name, int64_t action_id)
      : plugin_name(plugin_name), identifier(action_id) {}

  /// Unique name identifying the accelerator plugin.
  std::string plugin_name;
  /// Human-readable label for the accelerator target.
  std::string session_name;
  /// Unique identifier for this action within the plugin.
  int64_t identifier = 0;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorActions &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AcceleratorActions &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H
