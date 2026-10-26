//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 exception class
/// interface (4.13.2.)
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_EXCEPTION_HPP
#define _LIBSYCL___IMPL_EXCEPTION_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;
class exception_list;

namespace detail {
void addAsyncException(exception_list &, const std::exception_ptr &);
}

// int is used as the underlying type for consistency with std::error_code.
enum class errc : int {
  success = 0,
  runtime = 1,
  kernel = 2,
  accessor = 3,
  nd_range = 4,
  event = 5,
  kernel_argument = 6,
  build = 7,
  invalid = 8,
  memory_allocation = 9,
  platform = 10,
  profiling = 11,
  feature_not_supported = 12,
  kernel_not_supported = 13,
  backend_mismatch = 14,
};

/// Constructs an error code in sycl_category().
///
/// \param e is the SYCL error code value.
/// \return an error code with the value \p e in sycl_category().
_LIBSYCL_EXPORT std::error_code make_error_code(sycl::errc e) noexcept;

/// Obtains a reference to the static error category object for SYCL errors.
///
/// This object overrides the virtual function error_category::name() to return
/// a pointer to the string "sycl". When the implementation throws an
/// sycl::exception object Ex with this category, the error code value contained
/// by the exception (Ex.code().value()) is one of the enumerated values in
/// sycl::errc.
///
/// \return the error category object for SYCL errors.
_LIBSYCL_EXPORT const std::error_category &sycl_category() noexcept;

/// Reports a synchronous or asynchronous SYCL error in host code, with an error
/// code and an optional associated context.
///
/// Derived from std::exception so uncaught exceptions are printed in C++
/// default exception handler. Virtual inheritance is mandated by SYCL 2020.
///
/// \sa SYCL 2020 4.13.2. Exception class interface.
class _LIBSYCL_EXPORT exception : public virtual std::exception {
public:
  /// Constructs a SYCL exception with an error code and a message.
  ///
  /// \param ec is the error code.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(std::error_code ec, const char *what_arg)
      : exception(ec, nullptr, what_arg) {}

  /// \overload
  ///
  /// \param ec is the error code.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(std::error_code ec, const std::string &what_arg)
      : exception(ec, what_arg.c_str()) {}

  /// \overload
  ///
  /// \param ec is the error code.
  exception(std::error_code ec) : exception(ec, "") {}

  /// \overload
  ///
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(int ev, const std::error_category &ecat,
            const std::string &what_arg)
      : exception(ev, ecat, what_arg.c_str()) {}

  /// \overload
  ///
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(int ev, const std::error_category &ecat, const char *what_arg)
      : exception({ev, ecat}, what_arg) {}

  /// \overload
  ///
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  exception(int ev, const std::error_category &ecat)
      : exception({ev, ecat}, "") {}

  // To avoid cross-dependency issues between sycl::context and sycl::exception,
  // definition of ctors that require a context parameter are moved to
  // context.hpp.

  /// Constructs a SYCL exception with an associated SYCL context, an error code
  /// and a message.
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ec is the error code.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(context ctx, std::error_code ec, const std::string &what_arg);

  /// \overload
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ec is the error code.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(context ctx, std::error_code ec, const char *what_arg);

  /// \overload
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ec is the error code.
  exception(context ctx, std::error_code ec);

  /// \overload
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(context ctx, int ev, const std::error_category &ecat,
            const std::string &what_arg);

  /// \overload
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  /// \param what_arg is a message describing the error condition; the string
  /// returned by what() contains it as a substring.
  exception(context ctx, int ev, const std::error_category &ecat,
            const char *what_arg);

  /// \overload
  ///
  /// \param ctx is the SYCL context associated with the exception.
  /// \param ev is the error code.
  /// \param ecat is the underlying error category of \p ev.
  exception(context ctx, int ev, const std::error_category &ecat);

  virtual ~exception();

  /// \return the error code stored inside the exception.
  const std::error_code &code() const noexcept;

  /// \return the error category of the error code stored inside the exception.
  const std::error_category &category() const noexcept;

  /// \return an implementation-defined non-null constant C-style string that
  /// describes the error that triggered the exception.
  const char *what() const noexcept final;

  /// \return true if this SYCL exception has an associated SYCL context and
  /// false if it does not.
  bool has_context() const noexcept;

  /// Returns the SYCL context associated with this exception, if one is
  /// available.
  ///
  /// \return the SYCL context associated with this exception.
  /// \throw sycl::exception with sycl::errc::invalid if this exception does not
  /// have an associated context (has_context() == false).
  context get_context() const;

private:
  exception(std::error_code EC, std::shared_ptr<context> SharedPtrCtx,
            const char *WhatArg);
  // Exceptions must be noexcept copy constructible, so cannot use std::string
  // or context directly.
  std::shared_ptr<std::string> MMessage;
  std::shared_ptr<context> MContext;
  std::error_code MErrC;
};

/// Used as a container for a list of asynchronous exceptions.
///
/// \sa SYCL 2020 4.13.2. Exception class interface.
class _LIBSYCL_EXPORT exception_list {
public:
  using value_type = std::exception_ptr;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = std::vector<std::exception_ptr>::const_iterator;
  using const_iterator = std::vector<std::exception_ptr>::const_iterator;

  /// \return the size of the list.
  size_type size() const;

  /// \return an iterator to the beginning of the list of asynchronous
  /// exceptions.
  iterator begin() const;

  /// \return an iterator to the end of the list of asynchronous exceptions.
  iterator end() const;

private:
  std::vector<std::exception_ptr> MList;

  friend void detail::addAsyncException(exception_list &, const_reference);
};

namespace detail {
// Default implementation of async_handler used by queue and context when no
// user-defined async_handler is specified.
_LIBSYCL_EXPORT void defaultAsyncHandler(exception_list Exceptions);
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std

#endif // _LIBSYCL___IMPL_EXCEPTION_HPP
