//===--- MacroExpander.h - Format C++ code ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the main building blocks of macro support in
/// clang-format.
///
/// In order to not violate the requirement that clang-format can format files
/// in isolation, clang-format's macro support uses expansions users provide
/// as part of clang-format's style configuration.
///
/// Macro definitions are of the form "MACRO(p1, p2)=p1 + p2", but only support
/// one level of expansion (\see MacroExpander for a full description of what
/// is supported).
///
/// As part of parsing, clang-format uses the MacroExpander to expand the
/// spelled token streams into expanded token streams when it encounters a
/// macro call. The UnwrappedLineParser continues to parse UnwrappedLines
/// from the expanded token stream.
/// After the expanded unwrapped lines are parsed, the MacroUnexpander matches
/// the spelled token stream into unwrapped lines that best resemble the
/// structure of the expanded unwrapped lines.
///
/// When formatting, clang-format formats the expanded unwrapped lines first,
/// determining the token types. Next, it formats the spelled unwrapped lines,
/// keeping the token types fixed, while allowing other formatting decisions
/// to change.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_FORMAT_MACROS_H
#define CLANG_LIB_FORMAT_MACROS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "Encoding.h"
#include "FormatToken.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace clang {
class IdentifierTable;
class SourceManager;

namespace format {
struct FormatStyle;

/// Takes a set of macro definitions as strings and allows expanding calls to
/// those macros.
///
/// For example:
/// Definition: A(x, y)=x + y
/// Call      : A(int a = 1, 2)
/// Expansion : int a = 1 + 2
///
/// Expansion does not check arity of the definition.
/// If fewer arguments than expected are provided, the remaining parameters
/// are considered empty:
/// Call     : A(a)
/// Expansion: a +
/// If more arguments than expected are provided, they will be discarded.
///
/// The expander does not support:
/// - recursive expansion
/// - stringification
/// - concatenation
/// - variadic macros
///
/// Furthermore, only a single expansion of each macro argument is supported,
/// so that we cannot get conflicting formatting decisions from different
/// expansions.
/// Definition: A(x)=x+x
/// Call      : A(id)
/// Expansion : id+x
///
class MacroExpander {
public:
  using ArgsList = llvm::ArrayRef<llvm::SmallVector<FormatToken *, 8>>;

  /// Construct a macro expander from a set of macro definitions.
  /// Macro definitions must be encoded as UTF-8.
  ///
  /// Each entry in \p Macros must conform to the following simple
  /// macro-definition language:
  /// <definition> ::= <id> <expansion> | <id> "(" <params> ")" <expansion>
  /// <params>     ::= <id-list> | ""
  /// <id-list>    ::= <id> | <id> "," <params>
  /// <expansion>  ::= "=" <tail> | <eof>
  /// <tail>       ::= <tok> <tail> | <eof>
  ///
  /// Macros that cannot be parsed will be silently discarded.
  ///
  MacroExpander(const std::vector<std::string> &Macros,
                clang::SourceManager &SourceMgr, const FormatStyle &Style,
                llvm::SpecificBumpPtrAllocator<FormatToken> &Allocator,
                IdentifierTable &IdentTable);
  ~MacroExpander();

  /// Returns whether a macro \p Name is defined.
  bool defined(llvm::StringRef Name) const;

  /// Returns whether the macro has no arguments and should not consume
  /// subsequent parentheses.
  bool objectLike(llvm::StringRef Name) const;

  /// Returns the expanded stream of format tokens for \p ID, where
  /// each element in \p Args is a positional argument to the macro call.
  llvm::SmallVector<FormatToken *, 8> expand(FormatToken *ID,
                                             ArgsList Args) const;

private:
  struct Definition;
  class DefinitionParser;

  void parseDefinition(const std::string &Macro);

  clang::SourceManager &SourceMgr;
  const FormatStyle &Style;
  llvm::SpecificBumpPtrAllocator<FormatToken> &Allocator;
  IdentifierTable &IdentTable;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
  llvm::StringMap<Definition> Definitions;
};

} // namespace format
} // namespace clang

#endif
