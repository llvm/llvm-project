//===-- SymbolLocator.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_SYMBOLLOCATOR_H
#define LLDB_SYMBOL_SYMBOLLOCATOR_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/UUID.h"

namespace lldb_private {

/// A plug-in interface definition class for symbol locators.
///
/// Symbol locator plugins are responsible for finding and downloading debug
/// symbol files for binaries when they are not available locally. This is
/// particularly useful in environments where debug symbols are stored
/// separately from executables (e.g., symbol servers, build systems with
/// separate debug info repositories).
///
/// The main use case is automated symbol file discovery during debugging
/// sessions. When LLDB encounters a module with a specific UUID but cannot
/// find matching debug symbols locally, it can use SymbolLocator plugins to
/// attempt to locate and download the symbols from remote sources.
///
/// LLDB invokes SymbolLocator functionality through the PluginManager when
/// symbols are needed but not found. The download process is asynchronous and
/// happens on background threads to avoid blocking the debugging session.
///
/// Plugins should implement:
/// - Standard PluginInterface methods (GetPluginName, etc.)
/// - Logic to locate symbol files based on UUID and other module metadata
/// - Download capabilities for fetching symbols from remote sources
///
/// Symbol locator plugins integrate with LLDB's symbol download settings
/// (eSymbolDownloadOff, eSymbolDownloadBackground, eSymbolDownloadForeground)
/// to control when and how symbols are fetched.
class SymbolLocator : public PluginInterface {
public:
  SymbolLocator() = default;

  /// Locate the symbol file for the given UUID on a background thread. This
  /// function returns immediately. Under the hood it uses the debugger's
  /// thread pool to call DownloadObjectAndSymbolFile. If a symbol file is
  /// found, this will notify all target which contain the module with the
  /// given UUID.
  static void DownloadSymbolFileAsync(const UUID &uuid);
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_SYMBOLLOCATOR_H
