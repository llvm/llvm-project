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
#include <string>
#include <system_error>
#include <vector>

namespace clang::ssaf {

/// Fluent API for constructing contextual errors.
///
/// ErrorBuilder allows building error messages with layered context
/// information. Context is added from innermost to outermost, and the final
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

  // Private constructor - only accessible via static factories.
  explicit ErrorBuilder(std::error_code EC) : Code(EC) {}

  // Helper: Format message and add to context stack.
  template <typename... Args>
  void addFormattedContext(const char *Fmt, Args &&...ArgVals) {
    std::string Message =
        llvm::formatv(Fmt, std::forward<Args>(ArgVals)...).str();
    ContextStack.push_back(std::move(Message));
  }

public:
  // Static factory: Create new error from error code and formatted message.
  template <typename... Args>
  static ErrorBuilder create(std::error_code EC, const char *Fmt,
                             Args &&...ArgVals) {
    ErrorBuilder Builder(EC);
    Builder.addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return Builder;
  }

  // Convenience overload for std::errc.
  template <typename... Args>
  static ErrorBuilder create(std::errc EC, const char *Fmt, Args &&...ArgVals) {
    return create(std::make_error_code(EC), Fmt,
                  std::forward<Args>(ArgVals)...);
  }

  // Static factory: Wrap existing error and optionally add context.
  static ErrorBuilder wrap(llvm::Error E);

  // Add context (plain string).
  ErrorBuilder &context(const char *Msg);

  // Add context (formatted string).
  template <typename... Args>
  ErrorBuilder &context(const char *Fmt, Args &&...ArgVals) {
    addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return *this;
  }

  // Build the final error.
  llvm::Error build();
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_ERRORBUILDER_H
