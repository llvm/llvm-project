//===-- RegularExpression.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegularExpression_h_
#define liblldb_RegularExpression_h_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"

namespace lldb_private {

class RegularExpression {
public:
  /// Default constructor.
  ///
  /// The default constructor that initializes the object state such that it
  /// contains no compiled regular expression.
  RegularExpression() = default;

  explicit RegularExpression(llvm::StringRef string);
  ~RegularExpression() = default;

  RegularExpression(const RegularExpression &rhs);
  RegularExpression(RegularExpression &&rhs) = default;

  RegularExpression &operator=(RegularExpression &&rhs) = default;
  RegularExpression &operator=(const RegularExpression &rhs) = default;

  /// Compile a regular expression.
  ///
  /// Compile a regular expression using the supplied regular expression text.
  /// The compiled regular expression lives in this object so that it can be
  /// readily used for regular expression matches. Execute() can be called
  /// after the regular expression is compiled. Any previously compiled
  /// regular expression contained in this object will be freed.
  ///
  /// \param[in] re
  ///     A NULL terminated C string that represents the regular
  ///     expression to compile.
  ///
  /// \return \b true if the regular expression compiles successfully, \b false
  ///     otherwise.
  bool Compile(llvm::StringRef string);

  /// Executes a regular expression.
  ///
  /// Execute a regular expression match using the compiled regular expression
  /// that is already in this object against the match string \a s. If any
  /// parens are used for regular expression matches \a match_count should
  /// indicate the number of regmatch_t values that are present in \a
  /// match_ptr.
  ///
  /// \param[in] string
  ///     The string to match against the compile regular expression.
  ///
  /// \param[in] match
  ///     A pointer to a RegularExpression::Match structure that was
  ///     properly initialized with the desired number of maximum
  ///     matches, or nullptr if no parenthesized matching is needed.
  ///
  /// \return \b true if \a string matches the compiled regular expression, \b
  ///     false otherwise.
  bool Execute(llvm::StringRef string,
               llvm::SmallVectorImpl<llvm::StringRef> *matches = nullptr) const;

  /// Access the regular expression text.
  ///
  /// Returns the text that was used to compile the current regular
  /// expression.
  ///
  /// \return
  ///     The NULL terminated C string that was used to compile the
  ///     current regular expression
  llvm::StringRef GetText() const;

  /// Test if valid.
  ///
  /// Test if this object contains a valid regular expression.
  ///
  /// \return \b true if the regular expression compiled and is ready for
  ///     execution, \b false otherwise.
  bool IsValid() const;

  /// Return an error if the regular expression failed to compile.
  llvm::Error GetError() const;

private:
  /// A copy of the original regular expression text.
  std::string m_regex_text;
  /// The compiled regular expression.
  mutable llvm::Regex m_regex;
};

} // namespace lldb_private

#endif // liblldb_RegularExpression_h_
