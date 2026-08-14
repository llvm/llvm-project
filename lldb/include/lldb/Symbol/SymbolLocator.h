//===-- SymbolLocator.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_SYMBOLLOCATOR_H
#define LLDB_SYMBOL_SYMBOLLOCATOR_H

#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UUID.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <system_error>

namespace lldb_private {

class SymbolLocator : public PluginInterface {
public:
  SymbolLocator() = default;

  /// A binary was not found and nothing could say why. Distinct from every
  /// other error so that a caller composing its own message for a plain miss
  /// cannot mistake a real failure for one.
  class NotFound : public llvm::ErrorInfo<NotFound> {
  public:
    static char ID;

    void log(llvm::raw_ostream &os) const override;
    std::error_code convertToErrorCode() const override;
  };

  /// One binary to search for.
  struct Request {
    /// What to look for.
    ModuleSpec module_spec;

    /// A platform that may know where the binary is. The locator plugins have
    /// no Platform of their own to consult.
    lldb::PlatformSP platform;

    /// Allow contacting an external symbol server when the local searches come
    /// up empty.
    bool external_lookup = false;
  };

  /// What a search found.
  struct Result {
    /// The binary, and its symbol file if there is one to be had.
    ModuleSpec module_spec;

    /// What an external symbol server had to say about the symbols, even
    /// though the binary itself was found. Recorded rather than reported, so
    /// that it reaches the user in the caller's order, and so has to be
    /// consumed even by a caller that does not report it.
    std::optional<llvm::Error> symbol_error;

    /// Where the time went, to be merged into the Module once it exists.
    StatisticsMap statistics;
  };

  /// Find a binary and, if possible, its symbols.
  ///
  /// Mutates no Target and no Process, which is what lets a caller holding
  /// several binaries search for all of them before installing any.
  ///
  /// \return
  ///     Where the binary is, or an error if it could not be found at all.
  ///     Finding the binary but not its symbols is a success, with
  ///     Result::symbol_error carrying whatever a symbol server had to say
  ///     about them.
  ///
  ///     A miss carries an external symbol server's explanation if it gave
  ///     one, and is a NotFound when nothing could say why.
  static llvm::Expected<Result> Locate(const Request &request,
                                       const FileSpecList &search_paths);

  /// Locate the symbol file for the given UUID on a background thread. This
  /// function returns immediately. Under the hood it uses the debugger's
  /// thread pool to call DownloadObjectAndSymbolFile. If a symbol file is
  /// found, this will notify all target which contain the module with the
  /// given UUID.
  static void DownloadSymbolFileAsync(const UUID &uuid);
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_SYMBOLLOCATOR_H
