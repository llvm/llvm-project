//===-- TelemetryVendor.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_TELEMETRYVENDOR_H
#define LLDB_CORE_TELEMETRYVENDOR_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/Telemetry.h"
#include "llvm/Telemetry/Telemetry.h"

#include <memory>

namespace lldb_private {

class TelemetryVendor : public PluginInterface {
public:
  TelemetryVendor() = default;

  llvm::StringRef GetPluginName() override;

  static void Initialize();

  static void Terminate();

  static lldb::TelemetryConfig GetTelemetryConfig();

  static void SetTelemetryConfig(const lldb::TelemetryConfigSP &config);

  static lldb::TelemetryManagerSP GetTelemetryManager();

  static void SetTelemetryManager(const lldb::TelemetryManagerSP &manager_sp);
};

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRYVENDOR_H
