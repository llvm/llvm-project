//===-- Highlighter.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Highlighter.h"

#include "lldb/Target/Language.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/StreamString.h"
#include <optional>

using namespace lldb_private;
using namespace lldb_private::ansi;

void HighlightStyle::ColorStyle::Apply(Stream &s, llvm::StringRef value) const {
  s << m_prefix << value << m_suffix;
}

void HighlightStyle::ColorStyle::Set(llvm::StringRef prefix,
                                     llvm::StringRef suffix) {
  m_prefix = FormatAnsiTerminalCodes(prefix);
  m_suffix = FormatAnsiTerminalCodes(suffix);
}

static HighlightStyle::ColorStyle GetColor(const char *c) {
  return HighlightStyle::ColorStyle(c, "${ansi.normal}");
}

HighlightStyle HighlightStyle::MakeVimStyle() {
  HighlightStyle result;
  result.comment = GetColor("${ansi.fg.purple}");
  result.scalar_literal = GetColor("${ansi.fg.red}");
  result.keyword = GetColor("${ansi.fg.green}");
  return result;
}

const Highlighter &
HighlighterManager::getHighlighterFor(lldb::LanguageType language_type,
                                      llvm::StringRef path) const {
  // The language may be able to provide a language type based on the path.
  if (Language *language =
          lldb_private::Language::FindPlugin(language_type, path))
    language_type = language->GetLanguageType();

  std::lock_guard<std::mutex> guard(m_mutex);
  auto it = m_highlighters.find(language_type);
  if (it != m_highlighters.end())
    return *it->second;

  uint32_t idx = 0;
  while (HighlighterCreateInstance create_instance =
             PluginManager::GetHighlighterCreateCallbackAtIndex(idx++)) {
    if (Highlighter *highlighter = create_instance(language_type))
      m_highlighters.try_emplace(language_type,
                                 std::unique_ptr<Highlighter>(highlighter));
  }

  assert(m_highlighters.contains(language_type) &&
         "we should always find the default highlighter");
  return *m_highlighters[language_type];
}

std::string Highlighter::Highlight(const HighlightStyle &options,
                                   llvm::StringRef line,
                                   std::optional<size_t> cursor_pos,
                                   llvm::StringRef previous_lines) const {
  StreamString s;
  Highlight(options, line, cursor_pos, previous_lines, s);
  s.Flush();
  return s.GetString().str();
}
