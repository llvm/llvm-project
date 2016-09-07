//===-- Error.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __DCError_h__
#define __DCError_h__
#if defined(__cplusplus)

#include "llvm/Support/DataTypes.h"

#include <cstdarg>
#include <cstdio>
#include <string>

#include "lldb/lldb-private.h"

namespace lldb_private {

class Log;

//----------------------------------------------------------------------
/// @class Error Error.h "lldb/Core/Error.h"
/// @brief An error handling class.
///
/// This class is designed to be able to hold any error code that can be
/// encountered on a given platform. The errors are stored as a value
/// of type Error::ValueType. This value should be large enough to hold
/// any and all errors that the class supports. Each error has an
/// associated type that is of type lldb::ErrorType. New types
/// can be added to support new error types, and architecture specific
/// types can be enabled. In the future we may wish to switch to a
/// registration mechanism where new error types can be registered at
/// runtime instead of a hard coded scheme.
///
/// All errors in this class also know how to generate a string
/// representation of themselves for printing results and error codes.
/// The string value will be fetched on demand and its string value will
/// be cached until the error is cleared of the value of the error
/// changes.
//----------------------------------------------------------------------
class Error {
public:
  //------------------------------------------------------------------
  /// Every error value that this object can contain needs to be able
  /// to fit into ValueType.
  //------------------------------------------------------------------
  typedef uint32_t ValueType;

  //------------------------------------------------------------------
  /// Default constructor.
  ///
  /// Initialize the error object with a generic success value.
  ///
  /// @param[in] err
  ///     An error code.
  ///
  /// @param[in] type
  ///     The type for \a err.
  //------------------------------------------------------------------
  Error();

  explicit Error(ValueType err, lldb::ErrorType type = lldb::eErrorTypeGeneric);

  explicit Error(const char *format, ...) __attribute__((format(printf, 2, 3)));

  Error(const Error &rhs);
  //------------------------------------------------------------------
  /// Assignment operator.
  ///
  /// @param[in] err
  ///     An error code.
  ///
  /// @return
  ///     A const reference to this object.
  //------------------------------------------------------------------
  const Error &operator=(const Error &rhs);

  //------------------------------------------------------------------
  /// Assignment operator from a kern_return_t.
  ///
  /// Sets the type to \c MachKernel and the error code to \a err.
  ///
  /// @param[in] err
  ///     A mach error code.
  ///
  /// @return
  ///     A const reference to this object.
  //------------------------------------------------------------------
  const Error &operator=(uint32_t err);

  ~Error();

  //------------------------------------------------------------------
  /// Get the error string associated with the current error.
  //
  /// Gets the error value as a NULL terminated C string. The error
  /// string will be fetched and cached on demand. The error string
  /// will be retrieved from a callback that is appropriate for the
  /// type of the error and will be cached until the error value is
  /// changed or cleared.
  ///
  /// @return
  ///     The error as a NULL terminated C string value if the error
  ///     is valid and is able to be converted to a string value,
  ///     NULL otherwise.
  //------------------------------------------------------------------
  const char *AsCString(const char *default_error_str = "unknown error") const;

  //------------------------------------------------------------------
  /// Clear the object state.
  ///
  /// Reverts the state of this object to contain a generic success
  /// value and frees any cached error string value.
  //------------------------------------------------------------------
  void Clear();

  //------------------------------------------------------------------
  /// Test for error condition.
  ///
  /// @return
  ///     \b true if this object contains an error, \b false
  ///     otherwise.
  //------------------------------------------------------------------
  bool Fail() const;

  //------------------------------------------------------------------
  /// Access the error value.
  ///
  /// @return
  ///     The error value.
  //------------------------------------------------------------------
  ValueType GetError() const;

  //------------------------------------------------------------------
  /// Access the error type.
  ///
  /// @return
  ///     The error type enumeration value.
  //------------------------------------------------------------------
  lldb::ErrorType GetType() const;

  //------------------------------------------------------------------
  /// Log an error to Log().
  ///
  /// Log the error given a formatted string \a format. If the this
  /// object contains an error code, update the error string to
  /// contain the prefix "error: ", followed by the formatted string,
  /// followed by the error value and any string that describes the
  /// error value. This allows more context to be given to an error
  /// string that remains cached in this object. Logging always occurs
  /// even when the error code contains a non-error value.
  ///
  /// @param[in] format
  ///     A printf style format string.
  ///
  /// @param[in] ...
  ///     Variable arguments that are needed for the printf style
  ///     format string \a format.
  //------------------------------------------------------------------
  void PutToLog(Log *log, const char *format, ...)
      __attribute__((format(printf, 3, 4)));

  //------------------------------------------------------------------
  /// Log an error to Log() if the error value is an error.
  ///
  /// Log the error given a formatted string \a format only if the
  /// error value in this object describes an error condition. If the
  /// this object contains an error, update the error string to
  /// contain the prefix "error: " followed by the formatted string,
  /// followed by the error value and any string that describes the
  /// error value. This allows more context to be given to an error
  /// string that remains cached in this object.
  ///
  /// @param[in] format
  ///     A printf style format string.
  ///
  /// @param[in] ...
  ///     Variable arguments that are needed for the printf style
  ///     format string \a format.
  //------------------------------------------------------------------
  void LogIfError(Log *log, const char *format, ...)
      __attribute__((format(printf, 3, 4)));

  //------------------------------------------------------------------
  /// Set accessor from a kern_return_t.
  ///
  /// Set accesssor for the error value to \a err and the error type
  /// to \c MachKernel.
  ///
  /// @param[in] err
  ///     A mach error code.
  //------------------------------------------------------------------
  void SetMachError(uint32_t err);

  void SetExpressionError(lldb::ExpressionResults, const char *mssg);

  int SetExpressionErrorWithFormat(lldb::ExpressionResults, const char *format,
                                   ...) __attribute__((format(printf, 3, 4)));

  //------------------------------------------------------------------
  /// Set accesssor with an error value and type.
  ///
  /// Set accesssor for the error value to \a err and the error type
  /// to \a type.
  ///
  /// @param[in] err
  ///     A mach error code.
  ///
  /// @param[in] type
  ///     The type for \a err.
  //------------------------------------------------------------------
  void SetError(ValueType err, lldb::ErrorType type);

  //------------------------------------------------------------------
  /// Set the current error to errno.
  ///
  /// Update the error value to be \c errno and update the type to
  /// be \c Error::POSIX.
  //------------------------------------------------------------------
  void SetErrorToErrno();

  //------------------------------------------------------------------
  /// Set the current error to a generic error.
  ///
  /// Update the error value to be \c LLDB_GENERIC_ERROR and update the
  /// type to be \c Error::Generic.
  //------------------------------------------------------------------
  void SetErrorToGenericError();

  //------------------------------------------------------------------
  /// Set the current error string to \a err_str.
  ///
  /// Set accessor for the error string value for a generic errors,
  /// or to supply additional details above and beyond the standard
  /// error strings that the standard type callbacks typically
  /// provide. This allows custom strings to be supplied as an
  /// error explanation. The error string value will remain until the
  /// error value is cleared or a new error value/type is assigned.
  ///
  /// @param err_str
  ///     The new custom error string to copy and cache.
  //------------------------------------------------------------------
  void SetErrorString(const char *err_str);

  //------------------------------------------------------------------
  /// Set the current error string to a formatted error string.
  ///
  /// @param format
  ///     A printf style format string
  //------------------------------------------------------------------
  int SetErrorStringWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  int SetErrorStringWithVarArg(const char *format, va_list args);

  //------------------------------------------------------------------
  /// Test for success condition.
  ///
  /// Returns true if the error code in this object is considered a
  /// successful return value.
  ///
  /// @return
  ///     \b true if this object contains an value that describes
  ///     success (non-erro), \b false otherwise.
  //------------------------------------------------------------------
  bool Success() const;

  //------------------------------------------------------------------
  /// Test for a failure due to a generic interrupt.
  ///
  /// Returns true if the error code in this object was caused by an interrupt.
  /// At present only supports Posix EINTR.
  ///
  /// @return
  ///     \b true if this object contains an value that describes
  ///     failure due to interrupt, \b false otherwise.
  //------------------------------------------------------------------
  bool WasInterrupted() const;

protected:
  //------------------------------------------------------------------
  /// Member variables
  //------------------------------------------------------------------
  ValueType m_code;             ///< Error code as an integer value.
  lldb::ErrorType m_type;       ///< The type of the above error code.
  mutable std::string m_string; ///< A string representation of the error code.
};

} // namespace lldb_private

#endif // #if defined(__cplusplus)
#endif // #ifndef __DCError_h__
