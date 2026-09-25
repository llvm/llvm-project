//===-- ColorSetting.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_COLORSETTING_H
#define LLDB_UTILITY_COLORSETTING_H

#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class Stream;

/// A pair of ANSI terminal escape sequences used to colorize a piece of text.
///
/// The \c prefix is emitted immediately before the text and the \c suffix
/// immediately after it (typically a reset code) to restore the previous
/// terminal appearance.
class ColorSetting {
public:
  ColorSetting() = default;
  ColorSetting(llvm::StringRef prefix, llvm::StringRef suffix)
      : m_prefix(prefix), m_suffix(suffix) {}

  llvm::StringRef GetPrefix() const { return m_prefix; }
  llvm::StringRef GetSuffix() const { return m_suffix; }

  /// Write \a str to \a s wrapped in this setting's ANSI color codes.
  void render(Stream &s, llvm::StringRef str) const;

private:
  llvm::StringRef m_prefix;
  llvm::StringRef m_suffix;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_COLORSETTING_H
