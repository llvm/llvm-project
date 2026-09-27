//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_LAZYIMPORT_H
#define LLDB_HOST_WINDOWS_LAZYIMPORT_H

#include "lldb/Host/windows/windows.h"

namespace lldb_private {

template <typename FnPtr> class LazyImport {
public:
  LazyImport(const wchar_t *dll, const char *symbol)
      : m_resolved(Resolve(dll, symbol)) {}

  /// Returns the resolved function pointer, or nullptr if the DLL or symbol
  /// is unavailable on this system.
  FnPtr get() const { return m_resolved; }

  explicit operator bool() const { return m_resolved != nullptr; }
  FnPtr operator*() const { return m_resolved; }

private:
  static FnPtr Resolve(const wchar_t *dll, const char *symbol) {
    HMODULE module = ::LoadLibraryW(dll);
    if (!module)
      return nullptr;
    return reinterpret_cast<FnPtr>(
        reinterpret_cast<void *>(::GetProcAddress(module, symbol)));
  }

  FnPtr m_resolved;
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_LAZYIMPORT_H
