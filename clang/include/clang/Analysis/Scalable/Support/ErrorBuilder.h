//===- ErrorBuilder.h - Fluent API for contextual errors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ErrorBuilder class, which provides a fluent API for
//  constructing contextual error messages with layered context information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_ERRORBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_ERRORBUILDER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace clang::ssaf {

/// Fluent API for constructing contextual errors.
///
/// ErrorBuilder allows building error messages with layered context
/// information. Context is added innermost to outermost, and the final
/// error message presents the context in reverse order (outermost first).
///
/// Example usage:
///   return ErrorBuilder::create(std::errc::invalid_argument,
///                               "invalid value {0}", value)
///       .context("processing field '{0}'", fieldName)
///       .context("reading configuration")
///       .build();
class ErrorBuilder {
private:
  std::error_code Code;
  std::vector<std::string> ContextStack;

  explicit ErrorBuilder(std::error_code EC) : Code(EC) {}

  void pushContext(std::string Msg) {
    if (!Msg.empty()) {
      ContextStack.push_back(std::move(Msg));
    }
  }

  template <typename... Args>
  static std::string formatErrorMessage(const char *Fmt, Args &&...ArgVals) {
    return llvm::formatv(Fmt, std::forward<Args>(ArgVals)...).str();
  }

  template <typename... Args>
  void addFormattedContext(const char *Fmt, Args &&...ArgVals) {
    pushContext(formatErrorMessage(Fmt, std::forward<Args>(ArgVals)...));
  }

public:
  /// Create an ErrorBuilder with an error code and formatted message.
  ///
  /// \param EC The error code for this error.
  /// \param Fmt Format string for the error message (using llvm::formatv).
  /// \param ArgVals Arguments for the format string.
  /// \returns A new ErrorBuilder with the initial error message.
  ///
  /// Example:
  /// \code
  ///   return ErrorBuilder::create(std::errc::invalid_argument,
  ///                               "invalid value: {0}", 42)
  ///       .build();
  /// \endcode
  template <typename... Args>
  static ErrorBuilder create(std::error_code EC, const char *Fmt,
                             Args &&...ArgVals) {
    ErrorBuilder Builder(EC);
    Builder.addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return Builder;
  }

  /// Convenience overload that accepts std::errc instead of std::error_code.
  ///
  /// \param EC The error condition for this error.
  /// \param Fmt Format string for the error message.
  /// \param ArgVals Arguments for the format string.
  /// \returns A new ErrorBuilder with the initial error message.
  template <typename... Args>
  static ErrorBuilder create(std::errc EC, const char *Fmt, Args &&...ArgVals) {
    return create(std::make_error_code(EC), Fmt,
                  std::forward<Args>(ArgVals)...);
  }

  /// Wrap an existing error and optionally add context.
  ///
  /// Extracts the error code and message(s) from the given error. If multiple
  /// errors are joined (via llvm::joinErrors), their messages are combined
  /// using " + " separator.
  ///
  /// \param E The error to wrap. Must be a failure (cannot be success).
  /// \returns A new ErrorBuilder containing the wrapped error information.
  ///
  /// \pre E must evaluate to true (i.e., must be a failure). Wrapping
  ///      Error::success() is a programming error and will trigger an
  ///      assertion failure in debug builds.
  ///
  /// Example:
  /// \code
  ///   if (auto Err = foo())
  ///     return ErrorBuilder::wrap(std::move(Err))
  ///         .context("while processing file")
  ///         .build();
  /// \endcode
  static ErrorBuilder wrap(llvm::Error E);

  /// Add context information as a plain string.
  ///
  /// Empty strings are ignored and not added to the context stack.
  ///
  /// \param Msg Context message to add. Must be a null-terminated string.
  /// \returns Reference to this ErrorBuilder for method chaining.
  ///
  /// Example:
  /// \code
  ///   return ErrorBuilder::create(...)
  ///       .context("reading configuration file")
  ///       .build();
  /// \endcode
  ErrorBuilder &context(const char *Msg);

  /// Add context information with formatted string.
  ///
  /// Uses llvm::formatv for formatting. Empty messages (after formatting)
  /// are ignored and not added to the context stack.
  ///
  /// \param Fmt Format string (using llvm::formatv syntax).
  /// \param ArgVals Arguments for the format string.
  /// \returns Reference to this ErrorBuilder for method chaining.
  ///
  /// Example:
  /// \code
  ///   return ErrorBuilder::create(...)
  ///       .context("processing field '{0}'", fieldName)
  ///       .context("at line {0}, column {1}", line, col)
  ///       .build();
  /// \endcode
  template <typename... Args>
  ErrorBuilder &context(const char *Fmt, Args &&...ArgVals) {
    addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return *this;
  }

  /// Build and return the final error.
  ///
  /// Constructs an llvm::Error with all accumulated context. The context
  /// is presented in reverse order: most recent context first, original
  /// error message last. Each context layer is separated by a newline.
  ///
  /// \returns An llvm::Error containing the error code and formatted message.
  ///          Even if no context was added (empty context stack), an error
  ///          with the stored error code is returned.
  ///
  /// Example output:
  /// \code
  ///   // ErrorBuilder::create(errc::invalid_argument, "value is 42")
  ///   //     .context("processing field 'age'")
  ///   //     .context("reading config")
  ///   //     .build();
  ///   //
  ///   // Produces:
  ///   // "reading config
  ///   //  processing field 'age'
  ///   //  value is 42"
  /// \endcode
  llvm::Error build() const;

  /// Report a fatal error with formatted message and terminate execution.
  ///
  /// Combines llvm::formatv and llvm::report_fatal_error. This is a static
  /// utility method for reporting unrecoverable errors that indicate bugs
  /// or corrupted data.
  ///
  /// \param Fmt Format string for the error message (using llvm::formatv).
  /// \param ArgVals Arguments for the format string.
  ///
  /// Example:
  /// \code
  ///   ErrorBuilder::fatal("Entity {0} with {1} linkage already exists",
  ///                       entityId, linkageType);
  /// \endcode
  template <typename... Args>
  [[noreturn]] static void fatal(const char *Fmt, Args &&...ArgVals) {
    llvm::report_fatal_error(llvm::StringRef(
        formatErrorMessage(Fmt, std::forward<Args>(ArgVals)...)));
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_ERRORBUILDER_H
