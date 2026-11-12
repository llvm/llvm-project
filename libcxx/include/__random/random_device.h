//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANDOM_RANDOM_DEVICE_H
#define _LIBCPP___RANDOM_RANDOM_DEVICE_H

#include <__config>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_HAS_RANDOM_DEVICE

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

class _LIBCPP_EXPORTED_FROM_ABI random_device {
#  ifdef _LIBCPP_USING_DEV_RANDOM
  int __f_;
#  elif !defined(_LIBCPP_ABI_NO_RANDOM_DEVICE_COMPATIBILITY_LAYOUT)
  // Apple platforms used to use the `_LIBCPP_USING_DEV_RANDOM` code path, and now
  // use `arc4random()` as of this comment. In order to avoid breaking the ABI, we
  // retain the same layout as before.
#    if defined(__APPLE__)
  [[__maybe_unused__]] int __padding_; // padding to fake the `__f_` field above
#    endif

  // ... vendors can add workarounds here if they switch to a different representation ...

#  endif

public:
  // types
  typedef unsigned result_type;

  // generator characteristics
  static _LIBCPP_CONSTEXPR const result_type _Min = 0;
  static _LIBCPP_CONSTEXPR const result_type _Max = 0xFFFFFFFFu;

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR result_type min() { return _Min; }
  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR result_type max() { return _Max; }

  // constructors
#  ifndef _LIBCPP_CXX03_LANG
  _LIBCPP_HIDE_FROM_ABI random_device() : random_device("/dev/urandom") {}
  explicit random_device(const string& __token);
#  else
  explicit random_device(const string& __token = "/dev/urandom");
#  endif
  ~random_device();

  // generating functions
  [[__nodiscard__]] result_type operator()();

  // property functions
  [[__nodiscard__]] double entropy() const _NOEXCEPT;

  random_device(const random_device&)  = delete;
  void operator=(const random_device&) = delete;
};

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_RANDOM_DEVICE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANDOM_RANDOM_DEVICE_H
