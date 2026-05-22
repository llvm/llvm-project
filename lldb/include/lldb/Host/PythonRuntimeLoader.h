//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_PYTHONRUNTIMELOADER_H
#define LLDB_HOST_PYTHONRUNTIMELOADER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

class PythonRuntimeLoader {
public:
  /// Make the Python runtime available in the current process so the
  /// ScriptInterpreterPython plugin's undefined Python symbols can resolve.
  ///
  /// On POSIX this dlopens libpython into the process via
  /// llvm::sys::DynamicLibrary::getPermanentLibrary, leaving Python's
  /// stable-ABI symbols visible in the global namespace. On Windows this
  /// configures the DLL search path so the plugin's delay-load thunks can
  /// find python3xx.dll.
  ///
  /// No-op when Python is already loaded in the current process (i.e. when
  /// LLDB is imported into a Python interpreter).
  ///
  /// Honors the LLDB_PYTHON_LIBRARY environment variable (full path to a
  /// libpython binary or framework Python file). Otherwise walks a
  /// platform-specific list of well-known locations (Xcode, Command Line
  /// Tools, /Library/Frameworks, /opt/homebrew, /usr/local on Darwin; SONAME
  /// variants on Linux).
  ///
  /// The first call drives the load; subsequent calls return the cached
  /// outcome. Returns success on no-op, on first successful load, or on
  /// builds without Python support. Returns an Error aggregating the
  /// per-candidate failures when no Python runtime can be located.
  static llvm::Error Load();

  /// Path of the Python runtime that was loaded, for diagnostics. Empty if
  /// Python was already in the process, if loading failed, or on builds
  /// without Python support. Triggers the load on first call, mirroring
  /// Load().
  static llvm::StringRef GetLoadedPath();
};

} // namespace lldb_private

#endif // LLDB_HOST_PYTHONRUNTIMELOADER_H
