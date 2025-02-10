//===-- TelemetryVendor.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/TelemetryVendor.h"

namespace lldb_private {

llvm::StringRef TelemetryVendor::GetPluginName() {
  return "UpstreamTelemetryVendor";
}

void TelemetryVendor::Initialize() {
  // The default (upstream) impl will have telemetry disabled by default.
  SetTelemetryConfig(
      std::make_unique<llvm::telemetry::Config>(/*enable_telemetry*/ false));
  SetTelemetryManager(nullptr);
}

static std::unique_ptr<llvm::telemetry::Config> current_config;
std::unique_ptr<llvm::telemetry::Config> TelemetryVendor::GetTelemetryConfig() {
  return current_config;
}

void TelemetryVendor::SetTelemetryConfig(
    std::unique_ptr<llvm::telemetry::Config> config) {
  current_config = std::move(config);
}

lldb::TelemetryManagerSP TelemetryVendor::GetTelemetryManager() {
  static TelemteryManagerSP g_telemetry_manager_sp;
  return g_telemetry_manager_sp;
}

void SetTelemetryManager(const lldb::TelemetryManagerSP &manager_sp) {
  GetTelemetryManager() = manager_sp;
}

} // namespace lldb_private
