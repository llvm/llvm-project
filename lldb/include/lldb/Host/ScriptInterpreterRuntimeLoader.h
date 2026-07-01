//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_SCRIPTINTERPRETERRUNTIMELOADER_H
#define LLDB_HOST_SCRIPTINTERPRETERRUNTIMELOADER_H

#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

/// Loads a script-interpreter runtime into the current process before its
/// plugin is dlopened. Lives outside the PluginManager because plugin
/// registration itself depends on the runtime's symbols already being
/// resolvable. Subclasses encode a single language's resolution policy
/// (search order, env-var overrides, platform candidates).
class ScriptInterpreterRuntimeLoader {
public:
  virtual ~ScriptInterpreterRuntimeLoader();

  /// Resolves the runtime so the script interpreter plugin's undefined
  /// symbols can bind. The first call drives the search and subsequent
  /// calls return the cached outcome. Returns success when the runtime
  /// is already in the process or has been loaded.
  virtual llvm::Error Load() = 0;

  /// Absolute path of the loaded runtime, for diagnostics. The success
  /// value is empty when the runtime was already in the process. Drives
  /// the load on first call.
  virtual llvm::Expected<llvm::StringRef> GetLoadedPath() = 0;

  /// True if the runtime is currently mapped into the process.
  virtual bool IsLoaded() = 0;

  /// Returns the loader for \p language. Returns an Error when the
  /// language has no dynamic loader (currently every language except
  /// Python) or when support for it was not compiled in.
  static llvm::Expected<ScriptInterpreterRuntimeLoader &>
  Get(lldb::ScriptLanguage language);
};

} // namespace lldb_private

#endif // LLDB_HOST_SCRIPTINTERPRETERRUNTIMELOADER_H
