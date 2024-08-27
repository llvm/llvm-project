//===-- Status.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_STATUS_H
#define LLDB_UTILITY_STATUS_H

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdarg>
#include <cstdint>
#include <string>
#include <system_error>
#include <type_traits>

namespace llvm {
class raw_ostream;
}

namespace lldb_private {

/// \class Status Status.h "lldb/Utility/Status.h" An error handling class.
///
/// This class is designed to be able to hold any error code that can be
/// encountered on a given platform. The errors are stored as a value of type
/// Status::ValueType. This value should be large enough to hold any and all
/// errors that the class supports. Each error has an associated type that is
/// of type lldb::ErrorType. New types can be added to support new error
/// types, and architecture specific types can be enabled. In the future we
/// may wish to switch to a registration mechanism where new error types can
/// be registered at runtime instead of a hard coded scheme.
///
/// All errors in this class also know how to generate a string representation
/// of themselves for printing results and error codes. The string value will
/// be fetched on demand and its string value will be cached until the error
/// is cleared of the value of the error changes.
class Status {
public:
  /// Every error value that this object can contain needs to be able to fit
  /// into ValueType.
  typedef uint32_t ValueType;

  Status();

  /// Initialize the error object with a generic success value.
  ///
  /// \param[in] err
  ///     An error code.
  ///
  /// \param[in] type
  ///     The type for \a err.
  explicit Status(ValueType err, lldb::ErrorType type = lldb::eErrorTypeGeneric,
                  std::string msg = {});

  Status(std::error_code EC);

  /// Create a generic error with the message \c err_str.
  explicit Status(std::string err_str);

  static Status FromErrorString(const char *str) {
    if (str)
      return Status(std::string(str));
    return Status(std::string("null error"));
  }

  static Status FromErrorStringWithFormat(const char *format, ...)
      __attribute__((format(printf, 1, 2)));

  template <typename... Args>
  static Status FromErrorStringWithFormatv(const char *format, Args &&...args) {
    return Status(llvm::formatv(format, std::forward<Args>(args)...));
  }

  static Status FromExpressionError(lldb::ExpressionResults result,
                                    std::string msg) {
    return Status(result, lldb::eErrorTypeExpression, msg);
  }

  ////////////////////////////////////////////////////////////////////////////
  // TO BE REMOVED ASAP.
  // This is transitional to make it easier to iterate with broken bots.
  ////////////////////////////////////////////////////////////////////////////
  LLVM_DEPRECATED("Use Status::FromErrorString() instead", "FromErrorString")
  explicit Status(const char *format, ...)
      __attribute__((format(printf, 2, 3))) {
    // Yes, this drops the arguments.
    *this = Status::FromErrorString(format);
  }
  template <typename... Args>
  static LLVM_DEPRECATED("Use Status::FromErrorStringWithFormat() instead",
                         "FromErrorStringWithFormat") Status
      createWithFormat(const char *format, Args &&...args) {
    return Status::FromErrorStringWithFormat(format,
                                             std::forward<Args>(args)...);
  }
  LLVM_DEPRECATED("Use Status::FromExpressionError() instead",
                  "FromExpressionError")
  void SetExpressionError(lldb::ExpressionResults results, const char *msg) {
    *this = Status::FromExpressionError(results, msg);
  }
  LLVM_DEPRECATED("Use Status::FromExpressionError() instead",
                  "FromExpressionError")
  int SetExpressionErrorWithFormat(lldb::ExpressionResults results,
                                   const char *msg, ...) {
    *this = Status::FromExpressionError(results, msg);
    return 0;
  }
  LLVM_DEPRECATED("Use Status::Status() instead", "Status")
  void SetError(ValueType err, lldb::ErrorType type) {
    Status error(err, lldb::eErrorTypeGeneric);
    *this = error;
  }
  LLVM_DEPRECATED("Use Status::FromErrNo() instead", "Status")
  void SetErrorToErrno() { *this = Status::FromErrno(); }
  LLVM_DEPRECATED("Use Status() instead", "Status")
  void SetErrorToGenericError() {
    *this = Status::FromErrorString("generic error");
  }
  LLVM_DEPRECATED("Use Status::FromErrorString() instead", "Status")
  void SetErrorString(llvm::StringRef err_str) {
    *this = Status::FromErrorString(err_str.str().c_str());
  }
  LLVM_DEPRECATED("Use Status::FromErrorStringWithFormat() instead", "Status")
  int SetErrorStringWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3))) {
    *this = Status::FromErrorString(format);
    return 0;
  }
  LLVM_DEPRECATED("Use Status::FromErrorString() instead", "Status")
  int SetErrorStringWithVarArg(const char *format, va_list args) {
    *this = Status::FromErrorString(format);
    return 0;
  }
  template <typename... Args>
  LLVM_DEPRECATED("Use Status::FromErrorStringWithFormatv() instead", "Status")
  void SetErrorStringWithFormatv(const char *format, Args &&...args) {
    *this =
        Status::FromErrorStringWithFormatv(format, std::forward<Args>(args)...);
  }
  ////////////////////////////////////////////////////////////////////////////

  /// Set the current error to errno.
  ///
  /// Update the error value to be \c errno and update the type to be \c
  /// Status::POSIX.
  static Status FromErrno();

  ~Status();

  // llvm::Error support
  explicit Status(llvm::Error error) { *this = std::move(error); }
  const Status &operator=(llvm::Error error);
  llvm::Error ToError() const;

  /// Get the error string associated with the current error.
  //
  /// Gets the error value as a NULL terminated C string. The error string
  /// will be fetched and cached on demand. The error string will be retrieved
  /// from a callback that is appropriate for the type of the error and will
  /// be cached until the error value is changed or cleared.
  ///
  /// \return
  ///     The error as a NULL terminated C string value if the error
  ///     is valid and is able to be converted to a string value,
  ///     NULL otherwise.
  const char *AsCString(const char *default_error_str = "unknown error") const;

  /// Clear the object state.
  ///
  /// Reverts the state of this object to contain a generic success value and
  /// frees any cached error string value.
  void Clear();

  /// Test for error condition.
  ///
  /// \return
  ///     \b true if this object contains an error, \b false
  ///     otherwise.
  bool Fail() const;

  /// Access the error value.
  ///
  /// \return
  ///     The error value.
  ValueType GetError() const;

  /// Access the error type.
  ///
  /// \return
  ///     The error type enumeration value.
  lldb::ErrorType GetType() const;

  /// Test for success condition.
  ///
  /// Returns true if the error code in this object is considered a successful
  /// return value.
  ///
  /// \return
  ///     \b true if this object contains an value that describes
  ///     success (non-erro), \b false otherwise.
  bool Success() const;

protected:
  /// Status code as an integer value.
  ValueType m_code = 0;
  /// The type of the above error code.
  lldb::ErrorType m_type = lldb::eErrorTypeInvalid;
  /// A string representation of the error code.
  mutable std::string m_string;
};

} // namespace lldb_private

namespace llvm {
template <> struct format_provider<lldb_private::Status> {
  static void format(const lldb_private::Status &error, llvm::raw_ostream &OS,
                     llvm::StringRef Options);
};
} // namespace llvm

#define LLDB_ERRORF(status, fmt, ...)                                          \
  do {                                                                         \
    if (status) {                                                              \
      (status)->SetErrorStringWithFormat((fmt), __VA_ARGS__);                  \
    }                                                                          \
  } while (0);

#endif // LLDB_UTILITY_STATUS_H
