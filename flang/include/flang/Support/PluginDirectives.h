//===-- include/flang/Support/PluginDirectives.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiler directives defined by plugins:
//
//   !DIR$ prefix keyword [ ( arg [, arg]... ) ] [ name ( value ) ]...
//   arg -> [ name = ] value
//   value -> name | /common-block/ | integer | [sign] real | character-literal
//
// A trailing `name(value)` is the same as a `name=value` argument.
//
// A plugin loaded with `flang -fc1 -load` registers the directives it defines
// from a static initializer. The parser accepts the form above only for a
// registered prefix; semantics resolves name arguments to symbols and checks
// them against the registered argument kinds; lowering attaches the resolved
// directive to its subject (a procedure or a variable) as an MLIR attribute,
// for the plugin's own passes to interpret.
//
// A directive whose subject is a loop goes in the execution part, in front of
// a DO or DO WHILE loop, as !DIR$ UNROLL does. Its positional arguments are
// variables (or COMMON blocks). Lowering evaluates them in front of the loop
// (an allocatable or pointer as its descriptor then) and passes them to a
// marker call at the start of the loop body:
//
//   fir.call @__flang_directive.prefix.keyword(%var...)
//       {fir.directive = {prefix = "...", keyword = "...", args = {...}}}
//
// whose `args` are the other arguments, as for `fir.directives`. The plugin's
// passes replace the call, which nothing defines, with what it stands for.
//
// A plugin may also register a comment sentinel for its prefix, so that
//
//   !$prefix keyword [ ( arg [, arg]... ) ]
//
// is the same directive, and other compilers see a comment.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_PLUGINDIRECTIVES_H_
#define FORTRAN_SUPPORT_PLUGINDIRECTIVES_H_

#include <string>
#include <string_view>
#include <vector>

namespace Fortran::common {

/// What a directive argument must be.
enum class PluginDirectiveArgKind {
  Procedure, ///< A name resolving to a procedure.
  Variable, ///< A name resolving to a variable.
  Integer, ///< An integer literal.
  Real, ///< A real or integer literal, possibly signed.
  String, ///< A character literal, or a name taken as its spelling.
};

struct PluginDirectiveArg {
  /// Empty for a positional argument.
  std::string keyword;
  PluginDirectiveArgKind kind;
  bool required{false};
};

/// What a directive applies to.
enum class PluginDirectiveSubject {
  /// A procedure: the first positional argument if it is given, otherwise
  /// the subprogram whose specification part holds the directive.
  Procedure,
  /// A variable: the first positional argument.
  Variable,
  /// Either, as for Procedure.
  Any,
  /// The DO or DO WHILE loop that follows the directive. The positional
  /// arguments are variables or COMMON blocks.
  Loop,
};

struct PluginDirectiveSpec {
  std::string prefix; ///< Lower case, e.g. "enzyme".
  std::string keyword; ///< Lower case, e.g. "custom_rule".
  PluginDirectiveSubject subject{PluginDirectiveSubject::Procedure};
  /// The arguments after the (optional) positional subject.
  std::vector<PluginDirectiveArg> args;
  /// For a Loop directive, the least number of positional arguments.
  unsigned minPositional{0};
};

/// Register a directive. Call from a static initializer in a plugin.
void registerPluginDirective(PluginDirectiveSpec spec);

/// Whether a plugin registered a directive with this (lower case) prefix.
bool isPluginDirectivePrefix(std::string_view prefix);

/// The registered directive, or null.
const PluginDirectiveSpec *lookupPluginDirective(
    std::string_view prefix, std::string_view keyword);

/// Make `!$prefix` a directive sentinel for the directives with this (lower
/// case) prefix: in free form, and in fixed form with `!`, `c` or `*` in
/// column 1. The prescanner spells such a line `!dir$ prefix ...`, so it is
/// parsed, resolved, written to module files and lowered as that spelling.
/// A fixed form sentinel longer than four characters ends where `$prefix`
/// does, and the column after it takes the place of column 6: blank on an
/// initial line, and the continuation mark on a continuation line.
/// `!$` followed by a blank remains OpenMP conditional compilation.
/// Call from a static initializer in a plugin, like registerPluginDirective.
void registerPluginDirectiveSentinel(std::string_view prefix);

/// The sentinels registered by registerPluginDirectiveSentinel, as "$prefix".
const std::vector<std::string> &getPluginDirectiveSentinels();

} // namespace Fortran::common

#endif // FORTRAN_SUPPORT_PLUGINDIRECTIVES_H_
