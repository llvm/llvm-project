//===-- TelemetryVendor.h ---------------------------------------*- C++ -*-===//
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
  static TelemetryVendor *FindPlugin();

  TelemetryVendor() = default;

  llvm::StringRef GetPluginName() override;

  std::unique_ptr<llvm::telemetry::Config> GetTelemetryConfig();

  // Creates an LldbTelemeter instance.
  // Vendor plugins can override this to create customized instance as needed.
  virtual std::unique_ptr<LldbTelemeter>
  CreateTelemeter(lldb_private::Debugger *debugger);

  // TODO: move most of the basictelemeter concrete impl here to the plug in (to
  // its .ccpp file that is)
protected:
  // Returns a vendor-specific config which may or may not be the same as the
  // given "default_config". Downstream implementation can define their
  // configugrations in addition to OR overriding the default option.
  virtual std::unique_ptr<llvm::telemetry::Config> GetVendorSpecificConfig(
      std::unique_ptr<llvm::telemetry::Config> default_config);
};

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRYVENDOR_H
