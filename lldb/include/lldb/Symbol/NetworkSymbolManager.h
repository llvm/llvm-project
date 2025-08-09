//===-- NetworkSymbolManager.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_NETWORKSYMBOLMANAGER_H
#define LLDB_SYMBOL_NETWORKSYMBOLMANAGER_H

#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <mutex>

namespace lldb {
class SBDebugger;
}

namespace lldb_private {

/// NetworkSymbolManager provides centralized management of network-based
/// symbol loading optimizations, including server availability caching,
/// adaptive timeout management, and intelligent fallback strategies.
///
/// This class implements the architectural separation between DAP protocol
/// handling and network symbol optimization logic, addressing reviewer
/// concerns about layer violations in PR #150777.
class NetworkSymbolManager {
public:
  /// Configuration for network symbol optimization
  struct Configuration {
    /// Enable intelligent server availability caching
    bool enable_server_caching = true;

    /// Default timeout for debuginfod requests (milliseconds)
    uint32_t debuginfod_timeout_ms = 2000;

    /// Default timeout for symbol server requests (milliseconds)
    uint32_t symbol_server_timeout_ms = 2000;

    /// Completely disable network symbol loading
    bool disable_network_symbols = false;

    /// Enable adaptive timeout adjustment based on server response history
    bool enable_adaptive_timeouts = true;

    /// Cache TTL for server availability information (minutes)
    uint32_t cache_ttl_minutes = 5;
  };

  /// Server availability information with response time tracking
  struct ServerAvailability {
    bool is_responsive = false;
    std::chrono::steady_clock::time_point last_checked;
    std::chrono::milliseconds average_response_time{0};
    uint32_t success_count = 0;
    uint32_t failure_count = 0;
    uint32_t consecutive_failures = 0;
    std::chrono::steady_clock::time_point first_failure_time;

    ServerAvailability() : last_checked(std::chrono::steady_clock::now()) {}

    /// Calculate reliability score (0.0 = unreliable, 1.0 = highly reliable)
    double GetReliabilityScore() const;

    /// Check if cached information is still valid
    bool IsValid(std::chrono::minutes ttl) const;

    /// Check if server should be temporarily blacklisted due to consecutive failures
    bool IsTemporarilyBlacklisted() const;

    /// Get recommended backoff time before next attempt
    std::chrono::minutes GetBackoffTime() const;
  };

  NetworkSymbolManager();
  ~NetworkSymbolManager();

  /// Configure network symbol optimization settings.
  /// This method respects existing user settings and provides opt-in behavior.
  Status Configure(const Configuration &config,
                   bool respect_user_settings = true);

  /// Get current configuration
  const Configuration &GetConfiguration() const { return config_; }

  /// Test server availability with intelligent caching.
  /// Returns cached result if available and valid, otherwise performs test.
  bool IsServerResponsive(
      llvm::StringRef server_url,
      std::chrono::milliseconds test_timeout = std::chrono::milliseconds(1000));

  /// Get adaptive timeout for a specific server based on response history.
  std::chrono::milliseconds GetAdaptiveTimeout(
      llvm::StringRef server_url) const;

  /// Record server response for adaptive timeout calculation.
  void RecordServerResponse(llvm::StringRef server_url,
                            std::chrono::milliseconds response_time,
                            bool success);

  /// Clear all cached server availability information
  void ClearServerCache();

  /// Apply network symbol optimizations to LLDB settings.
  /// This method queries existing settings before making changes.
  Status ApplyOptimizations(Debugger &debugger);

  /// Apply optimizations using SBDebugger interface (for DAP layer)
  Status ApplyOptimizations(lldb::SBDebugger &debugger);

  /// Restore original LLDB settings (for cleanup or user preference changes).
  Status RestoreOriginalSettings(Debugger &debugger);

  /// Restore settings using SBDebugger interface (for DAP layer)
  Status RestoreOriginalSettings(lldb::SBDebugger &debugger);

  /// Check if network symbol loading should be disabled based on configuration
  bool ShouldDisableNetworkSymbols() const;

  /// Get recommended timeout for debuginfod based on server availability
  std::chrono::milliseconds GetRecommendedDebuginfodTimeout() const;

  /// Get recommended timeout for symbol servers based on server availability
  std::chrono::milliseconds GetRecommendedSymbolServerTimeout() const;

  /// Attempt symbol resolution with intelligent fallback strategies.
  /// Returns true if symbols should be attempted from network, false if should
  /// skip.
  bool ShouldAttemptNetworkSymbolResolution(
      llvm::StringRef server_url) const;

  /// Get list of responsive servers for symbol resolution.
  std::vector<std::string> GetResponsiveServers(
      llvm::ArrayRef<llvm::StringRef> server_urls) const;

  /// Validate configuration parameters
  static Status ValidateConfiguration(const Configuration &config);

private:
  /// Current configuration
  Configuration config_;

  /// Server availability cache with thread safety
  mutable std::mutex server_cache_mutex_;
  llvm::StringMap<ServerAvailability> server_availability_cache_;

  /// Original LLDB settings for restoration
  mutable std::mutex settings_mutex_;
  llvm::StringMap<std::string> original_settings_;
  bool settings_applied_ = false;

  /// Test server connectivity (implementation detail)
  bool TestServerConnectivity(llvm::StringRef server_url,
                             std::chrono::milliseconds timeout);

  /// Query existing LLDB setting value
  Status QueryExistingSetting(Debugger &debugger,
                             llvm::StringRef setting_name,
                             std::string &value);

  /// Apply single LLDB setting with backup
  Status ApplySetting(Debugger &debugger,
                     llvm::StringRef setting_name,
                     llvm::StringRef value);
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_NETWORKSYMBOLMANAGER_H
