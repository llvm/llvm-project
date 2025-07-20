// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
  using std::clock_t _LIBCPP_USING_IF_EXISTS;
  using std::size_t _LIBCPP_USING_IF_EXISTS;
  using std::time_t _LIBCPP_USING_IF_EXISTS;

  using std::timespec _LIBCPP_USING_IF_EXISTS;
  using std::tm _LIBCPP_USING_IF_EXISTS;

  using std::asctime _LIBCPP_USING_IF_EXISTS;
  using std::clock _LIBCPP_USING_IF_EXISTS;
  using std::strftime _LIBCPP_USING_IF_EXISTS;

#ifndef _LIBCPP_ABI_VCRUNTIME
  using std::ctime _LIBCPP_USING_IF_EXISTS;
  using std::difftime _LIBCPP_USING_IF_EXISTS;
  using std::gmtime _LIBCPP_USING_IF_EXISTS;
  using std::localtime _LIBCPP_USING_IF_EXISTS;
  using std::mktime _LIBCPP_USING_IF_EXISTS;
  using std::time _LIBCPP_USING_IF_EXISTS;
  using std::timespec_get _LIBCPP_USING_IF_EXISTS;
#endif

// A workaround for UCRT because it defines these functions
// as static and that causes the error "internal linkage cannot be exported"
#ifdef _LIBCPP_ABI_VCRUNTIME

  template <int = 0>
  inline char* __CRTDECL ctime(_In_ time_t const* const _Time) noexcept {
    return _ctime64(_Time);
  }

  template <int = 0>
  inline double __CRTDECL difftime(_In_ time_t const _Time1, _In_ time_t const _Time2) noexcept {
    return _difftime64(_Time1, _Time2);
  }

  template <int = 0>
  inline tm* __CRTDECL gmtime(_In_ time_t const* const _Time) noexcept {
    return _gmtime64(_Time);
  }

  template <int = 0>
  inline tm* __CRTDECL localtime(_In_ time_t const* const _Time) noexcept {
    return _localtime64(_Time);
  }

  template <int = 0>
  inline time_t __CRTDECL mktime(_Inout_ tm* const _Tm) noexcept {
    return _mktime64(_Tm);
  }

  template <int = 0>
  inline time_t __CRTDECL time(_Out_opt_ time_t* const _Time) noexcept {
    return _time64(_Time);
  }

  template <int = 0>
  inline int __CRTDECL timespec_get(_Out_ timespec* const _Ts, _In_ int const _Base) noexcept {
    return _timespec64_get(reinterpret_cast<_timespec64*>(_Ts), _Base);
  }
#endif
} // namespace std
