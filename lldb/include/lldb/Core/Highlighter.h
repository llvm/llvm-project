//===-- Highlighter.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_HIGHLIGHTER_H
#define LLDB_CORE_HIGHLIGHTER_H

#include <optional>
#include <utility>
#include <vector>

#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

/// Represents style that the highlighter should apply to the given source code.
/// Stores information about how every kind of token should be annotated.
struct HighlightStyle {

  /// A pair of strings that should be placed around a certain token. Usually
  /// stores color codes in these strings (the suffix string is often used for
  /// resetting the terminal attributes back to normal).
  class ColorStyle {
    std::string m_prefix;
    std::string m_suffix;

  public:
    ColorStyle() = default;
    ColorStyle(llvm::StringRef prefix, llvm::StringRef suffix) {
      Set(prefix, suffix);
    }

    /// Applies this style to the given value.
    /// \param s
    ///     The stream to which the result should be appended.
    /// \param value
    ///     The value that we should place our strings around.
    void Apply(Stream &s, llvm::StringRef value) const;

    /// Sets the prefix and suffix strings.
    void Set(llvm::StringRef prefix, llvm::StringRef suffix);

    explicit operator bool() const {
      return !m_prefix.empty() && !m_suffix.empty();
    }
  };

  /// The style for the token which is below the cursor of the user. Note that
  /// this style is overwritten by the SourceManager with the values of
  /// stop-show-column-ansi-prefix/stop-show-column-ansi-suffix.
  ColorStyle selected;

  /// Matches identifiers to variable or functions.
  ColorStyle identifier;
  /// Matches any string or character literals in the language: "foo" or 'f'
  ColorStyle string_literal;
  /// Matches scalar value literals like '42' or '0.1'.
  ColorStyle scalar_literal;
  /// Matches all reserved keywords in the language.
  ColorStyle keyword;
  /// Matches any comments in the language.
  ColorStyle comment;
  /// Matches commas: ','
  ColorStyle comma;
  /// Matches one colon: ':'
  ColorStyle colon;
  /// Matches any semicolon: ';'
  ColorStyle semicolons;
  /// Matches operators like '+', '-', '%', '&', '='
  ColorStyle operators;

  /// Matches '{' or '}'
  ColorStyle braces;
  /// Matches '[' or ']'
  ColorStyle square_brackets;
  /// Matches '(' or ')'
  ColorStyle parentheses;

  // C language specific options

  /// Matches directives to a preprocessor (if the language has any).
  ColorStyle pp_directive;

  /// Returns a HighlightStyle that is based on vim's default highlight style.
  static HighlightStyle MakeVimStyle();
};

/// Annotates source code with color attributes.
///
/// Highlighter plugins provide syntax highlighting for source code displayed
/// in the debugger. These plugins apply ANSI color codes and formatting to
/// source text based on the programming language, making code easier to read
/// in the terminal or IDE.
///
/// LLDB uses highlighters in several contexts:
/// - Source code listings (source list command, frame info with source)
/// - Expression input in the REPL (interactive expression mode)
/// - Disassembly with interleaved source code
/// - Error messages showing problematic source lines
///
/// The HighlighterManager is responsible for selecting the appropriate
/// highlighter for a given language and file. When LLDB needs to display
/// source code, it calls HighlighterManager::getHighlighterFor() with the
/// LanguageType and file path. The manager:
/// 1. First attempts to determine the language from the Language plugin
/// 2. Checks if a highlighter for this language is already cached
/// 3. If not cached, queries all registered highlighter plugins via their
///    CreateInstance callbacks, passing the LanguageType
/// 4. Caches the result for future use
/// 5. Falls back to DefaultHighlighter if no language-specific highlighter
///    is available
///
/// Available highlighter implementations include:
/// - ClangHighlighter: Uses Clang's lexer for C/C++/Objective-C
/// - TreeSitterHighlighter: Uses tree-sitter parsers for multiple languages
/// - DefaultHighlighter: No-op highlighter that returns unmodified text
///
/// Key methods that subclasses must implement:
/// - GetName(): Returns a human-readable name for the highlighter
/// - Highlight(): The main method that applies syntax highlighting to a line
///   of code, considering the cursor position and previous lines for context
///
/// The Highlight() method receives:
/// - options: A HighlightStyle containing color codes for different token types
/// - line: The current line to highlight
/// - cursor_pos: Optional cursor position for highlighting the current token
/// - previous_lines: Context from earlier lines (useful for multi-line constructs)
/// - s: Output stream where highlighted text should be written
///
/// Implementations should:
/// - Parse the input text according to the language's syntax rules
/// - Apply appropriate ColorStyle from the HighlightStyle for each token type
///   (keywords, identifiers, literals, comments, operators, etc.)
/// - Handle multi-line constructs like block comments or string literals by
///   examining previous_lines
/// - Optionally highlight the token under the cursor differently
/// - Write the highlighted output to the stream, wrapping tokens with ANSI
///   color codes
/// - Be resilient to malformed or incomplete code (highlighting is often applied
///   to code being actively edited or partial expressions)
class Highlighter : public PluginInterface {
public:
  Highlighter() = default;
  virtual ~Highlighter() = default;
  Highlighter(const Highlighter &) = delete;
  const Highlighter &operator=(const Highlighter &) = delete;

  /// Returns a human readable name for the selected highlighter.
  virtual llvm::StringRef GetName() const = 0;

  /// Highlights the given line
  /// \param options
  ///     The highlight options.
  /// \param line
  ///     The user supplied line that needs to be highlighted.
  /// \param cursor_pos
  ///     The cursor position of the user in this line, starting at 0 (which
  ///     means the cursor is on the first character in 'line').
  /// \param previous_lines
  ///     Any previous lines the user has written which we should only use
  ///     for getting the context of the Highlighting right.
  /// \param s
  ///     The stream to which the highlighted version of the user string should
  ///     be written.
  virtual void Highlight(const HighlightStyle &options, llvm::StringRef line,
                         std::optional<size_t> cursor_pos,
                         llvm::StringRef previous_lines, Stream &s) const = 0;

  /// Utility method for calling Highlight without a stream.
  std::string Highlight(const HighlightStyle &options, llvm::StringRef line,
                        std::optional<size_t> cursor_pos,
                        llvm::StringRef previous_lines = "") const;
};

/// Manages the available highlighters.
///
/// HighlighterManager acts as a factory and cache for Highlighter plugins.
/// It maintains a map of language types to highlighter instances and lazily
/// instantiates highlighters on first use. This singleton-like manager ensures
/// that only one highlighter instance exists per language type, avoiding the
/// overhead of repeatedly creating highlighters for frequently displayed
/// languages.
///
/// The manager is thread-safe and uses a mutex to protect concurrent access
/// to the highlighter cache. When multiple threads request highlighters
/// simultaneously, only the first request for a given language will create
/// the highlighter instance, and subsequent requests will reuse the cached
/// instance.
class HighlighterManager {
public:
  /// Queries all known highlighter for one that can highlight some source code.
  ///
  /// \param language_type
  ///     The language type that the caller thinks the source code was given in.
  /// \param path
  ///     The path to the file the source code is from. Used as a fallback when
  ///     the user can't provide a language.
  /// \return
  ///     The highlighter that wants to highlight the source code. Could be an
  ///     empty highlighter that does nothing.
  const Highlighter &getHighlighterFor(lldb::LanguageType language_type,
                                       llvm::StringRef path) const;

private:
  mutable std::mutex m_mutex;
  mutable llvm::DenseMap<lldb::LanguageType, std::unique_ptr<Highlighter>>
      m_highlighters;
};

} // namespace lldb_private

namespace llvm {

/// DenseMapInfo implementation.
/// \{
template <> struct DenseMapInfo<lldb::LanguageType> {
  static inline lldb::LanguageType getEmptyKey() {
    return lldb::eNumLanguageTypes;
  }
  static inline lldb::LanguageType getTombstoneKey() {
    return lldb::eNumLanguageTypes;
  }
  static unsigned getHashValue(lldb::LanguageType language_type) {
    return static_cast<unsigned>(language_type);
  }
  static bool isEqual(lldb::LanguageType LHS, lldb::LanguageType RHS) {
    return LHS == RHS;
  }
};
/// \}

} // namespace llvm

#endif // LLDB_CORE_HIGHLIGHTER_H
