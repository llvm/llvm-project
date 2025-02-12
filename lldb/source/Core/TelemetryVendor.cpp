//===-- TelemetryVendor.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/llvm-config.h"

#ifdef LLVM_BUILD_TELEMETRY

#include "lldb/Core/TelemetryVendor.h"

namespace lldb_private {

static lldb::TelemetryConfigUP g_config_up =
    std::make_unique<llvm::telemetry::Config>(/*enable_telemetry*/ false);
lldb::TelemetryConfig *TelemetryVendor::GetTelemetryConfig() {
  return g_config_up.get();
}

static lldb::TelemteryManagerUP g_telemetry_manager_up = nullptr;
lldb::TelemetryManagerSP TelemetryVendor::GetTelemetryManager() {
  return g_telemetry_manager_sp.get();
}

void TelemetryVendor::SetTelemetryConfig(lldb::TelemetryConfigUP config) {
  g_config_up = std::move(config);
}

void SetTelemetryManager(lldb::TelemetryManagerUP &manager) {
  g_telemetry_manger_up = std::move(manager);
}

} // namespace lldb_private

#endif LLVM_BUILD_TELEMETRY
