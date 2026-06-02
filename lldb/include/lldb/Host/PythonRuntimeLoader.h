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
  /// Resolves the Python runtime so the script interpreter plugin's
  /// undefined symbols can bind. The first call drives the search and
  /// subsequent calls return the cached outcome. Returns success when
  /// Python is already in the process, when a runtime was loaded, or on
  /// builds without Python support. LLDB_PYTHON_LIBRARY overrides the
  /// default search.
  static llvm::Error Load();

  /// Absolute path of the loaded Python runtime, for diagnostics. Empty
  /// when the load failed, when Python was already in the process, or on
  /// builds without Python support. Drives the load on first call.
  static llvm::StringRef GetLoadedPath();

  /// True if libpython is currently mapped into the process. Distinguishes
  /// the lldb-in-python case from a build without Python support; both
  /// otherwise yield a successful Load() with an empty GetLoadedPath().
  static bool IsLoaded();
};

} // namespace lldb_private

#endif // LLDB_HOST_PYTHONRUNTIMELOADER_H
