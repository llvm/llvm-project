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

const char *ExpressionResultAsCString(lldb::ExpressionResults result);

/// Going a bit against the spirit of llvm::Error,
/// lldb_private::Status need to store errors long-term and sometimes
/// copy them. This base class defines an interface for this
/// operation.
class CloneableError
    : public llvm::ErrorInfo<CloneableError, llvm::ErrorInfoBase> {
public:
  using llvm::ErrorInfo<CloneableError, llvm::ErrorInfoBase>::ErrorInfo;
  CloneableError() : ErrorInfo() {}
  virtual std::unique_ptr<CloneableError> Clone() const = 0;
  static char ID;
};

/// Common base class for all error-code errors.
class CloneableECError
    : public llvm::ErrorInfo<CloneableECError, CloneableError> {
public:
  using llvm::ErrorInfo<CloneableECError, CloneableError>::ErrorInfo;
  CloneableECError() = delete;
  CloneableECError(std::error_code ec) : ErrorInfo(), EC(ec) {}
  std::error_code convertToErrorCode() const override { return EC; }
  void log(llvm::raw_ostream &OS) const override { OS << EC.message(); }
  std::unique_ptr<CloneableError> Clone() const override;
  static char ID;

protected:
  std::error_code EC;
};

/// FIXME: Move these declarations closer to where they're used.
class MachKernelError
    : public llvm::ErrorInfo<MachKernelError, CloneableECError> {
public:
  using llvm::ErrorInfo<MachKernelError, CloneableECError>::ErrorInfo;
  MachKernelError(std::error_code ec) : ErrorInfo(ec) {}
  std::string message() const override;
  std::unique_ptr<CloneableError> Clone() const override;
  static char ID;
};

class Win32Error : public llvm::ErrorInfo<Win32Error, CloneableECError> {
public:
  using llvm::ErrorInfo<Win32Error, CloneableECError>::ErrorInfo;
  Win32Error(std::error_code ec, const llvm::Twine &msg = {}) : ErrorInfo(ec) {}
  std::string message() const override;
  std::unique_ptr<CloneableError> Clone() const override;
  static char ID;
};

class ExpressionError
    : public llvm::ErrorInfo<ExpressionError, CloneableECError> {
public:
  using llvm::ErrorInfo<ExpressionError, CloneableECError>::ErrorInfo;
  ExpressionError(std::error_code ec, std::string msg = {})
      : ErrorInfo(ec), m_string(msg) {}
  std::unique_ptr<CloneableError> Clone() const override;
  std::string message() const override { return m_string; }
  static char ID;

protected:
  std::string m_string;
};

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
///
/// API design notes:
///
/// Most APIs that currently vend a Status would be better served by
/// returning llvm::Expected<> instead. If possibles APIs should be
/// refactored to avoid Status. The only legitimate long-term uses of
/// Status are objects that need to store an error for a long time
/// (which should be questioned as a design decision, too).
///
/// Implementation notes:
///
/// Internally, Status stores an llvm::Error.
///   eErrorTypeInvalid
///   eErrorTypeGeneric      llvm::StringError
///   eErrorTypePOSIX        llvm::ECError
///   eErrorTypeMachKernel   MachKernelError
///   eErrorTypeExpression   llvm::ErrorList<ExpressionError>
///   eErrorTypeWin32        Win32Error

class Status {
public:
  /// into ValueType.
  typedef uint32_t ValueType;

  Status();
  Status(Status &&other) = default;

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
                                    std::string msg);

  /// Set the current error to errno.
  ///
  /// Update the error value to be \c errno and update the type to be \c
  /// Status::POSIX.
  static Status FromErrno();

  ~Status();

  const Status &operator=(Status &&);
  /// Avoid using this in new code. Migrate APIs to llvm::Expected instead.
  static Status FromError(llvm::Error error);

  /// FIXME: Replace this with a takeError() method.
  llvm::Error ToError() const;
  /// Don't call this function in new code. Instead, redesign the API
  /// to use llvm::Expected instead of Status.
  Status Clone() const { return Status(ToError()); }

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
  /// If the internally stored \ref llvm::Error is an \ref
  /// llvm::ErrorList then this returns the error value of the first
  /// error.
  ///
  /// \return
  ///     The error value.
  ValueType GetError() const;

  /// Access the error type.
  ///
  /// If the internally stored \ref llvm::Error is an \ref
  /// llvm::ErrorList then this returns the error value of the first
  /// error.
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
  Status(llvm::Error error) : m_error(std::move(error)) {}
  llvm::Error m_error;
  /// TODO: Replace this with just callling toString(m_error).
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
