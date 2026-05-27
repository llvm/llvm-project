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
  constexpr LazyImport(const wchar_t *dll, const char *symbol)
      : m_dll(dll), m_symbol(symbol) {}

  /// Returns the resolved function pointer, or nullptr if the DLL or symbol
  /// is unavailable on this system. Resolution happens once.
  FnPtr get() const {
    static FnPtr resolved = Resolve(m_dll, m_symbol);
    return resolved;
  }

  explicit operator bool() const { return get() != nullptr; }
  FnPtr operator*() const { return get(); }

private:
  static FnPtr Resolve(const wchar_t *dll, const char *symbol) {
    HMODULE module = ::LoadLibraryW(dll);
    if (!module)
      return nullptr;
    return reinterpret_cast<FnPtr>(
        reinterpret_cast<void *>(::GetProcAddress(module, symbol)));
  }

  const wchar_t *m_dll;
  const char *m_symbol;
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_LAZYIMPORT_H
