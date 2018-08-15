//===-- Highlighter.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Highlighter.h"

#include "lldb/Target/Language.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

void HighlightStyle::ColorStyle::Apply(Stream &s, llvm::StringRef value) const {
  s << m_prefix << value << m_suffix;
}

void HighlightStyle::ColorStyle::Set(llvm::StringRef prefix,
                                     llvm::StringRef suffix) {
  m_prefix = lldb_utility::ansi::FormatAnsiTerminalCodes(prefix);
  m_suffix = lldb_utility::ansi::FormatAnsiTerminalCodes(suffix);
}

void NoHighlighter::Highlight(const HighlightStyle &options,
                              llvm::StringRef line,
                              llvm::StringRef previous_lines, Stream &s) const {
  // We just forward the input to the output and do no highlighting.
  s << line;
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
  Language *language = lldb_private::Language::FindPlugin(language_type, path);
  if (language && language->GetHighlighter())
    return *language->GetHighlighter();
  return m_no_highlighter;
}

std::string Highlighter::Highlight(const HighlightStyle &options,
                                   llvm::StringRef line,
                                   llvm::StringRef previous_lines) const {
  StreamString s;
  Highlight(options, line, previous_lines, s);
  s.Flush();
  return s.GetString().str();
}
