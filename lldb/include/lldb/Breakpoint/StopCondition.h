//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_STOPCONDITION_H
#define LLDB_BREAKPOINT_STOPCONDITION_H

#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class StopCondition {
public:
  StopCondition() = default;
  StopCondition(std::string text,
                lldb::LanguageType language = lldb::eLanguageTypeUnknown)
      : m_language(language) {
    SetText(std::move(text));
  }

  explicit operator bool() const { return !m_text.empty(); }

  llvm::StringRef GetText() const { return m_text; }

  void SetText(std::string text) {
    static std::hash<std::string> hasher;
    m_text = std::move(text);
    m_hash = hasher(text);
  }

  size_t GetHash() const { return m_hash; }

  lldb::LanguageType GetLanguage() const { return m_language; }

  void SetLanguage(lldb::LanguageType language) { m_language = language; }

private:
  /// The condition to test.
  std::string m_text;

  /// Its hash, so that locations know when the condition is updated.
  size_t m_hash = 0;

  /// The language for this condition.
  lldb::LanguageType m_language = lldb::eLanguageTypeUnknown;
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_STOPCONDITION_H
