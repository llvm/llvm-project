// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::clock_t _LIBCPP_USING_IF_EXISTS;
  using ::size_t _LIBCPP_USING_IF_EXISTS;
  using ::time_t _LIBCPP_USING_IF_EXISTS;

  using ::timespec _LIBCPP_USING_IF_EXISTS;
  using ::tm _LIBCPP_USING_IF_EXISTS;

  using ::asctime _LIBCPP_USING_IF_EXISTS;
  using ::clock _LIBCPP_USING_IF_EXISTS;

// NOTE: UCRT before Windows SDK 10.0.26100.6901 defines the <ctime> functions
// with 'static' so we cannot export them, but later versions fixed it and now
// define a macro we can use to detect older UCRT without the fix and we use it
// to avoid exporting them in order to compile without errors
#if !defined(_MSC_VER) || (defined(_STATIC_INLINE_UCRT_FUNCTIONS) && _STATIC_INLINE_UCRT_FUNCTIONS == 0)
  using ::ctime _LIBCPP_USING_IF_EXISTS;
  using ::difftime _LIBCPP_USING_IF_EXISTS;
  using ::gmtime _LIBCPP_USING_IF_EXISTS;
  using ::localtime _LIBCPP_USING_IF_EXISTS;
  using ::mktime _LIBCPP_USING_IF_EXISTS;
  using ::strftime _LIBCPP_USING_IF_EXISTS;
  using ::time _LIBCPP_USING_IF_EXISTS;
  using ::timespec_get _LIBCPP_USING_IF_EXISTS;
#endif
} // export
