// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::clock_t;
  using ::size_t;
  using ::time_t;

  using ::timespec;
  using ::tm;

  using ::asctime;
  using ::clock;
  using ::ctime;
  using ::difftime;
  using ::gmtime;
  using ::localtime;
  using ::mktime;
  using ::strftime;
  using ::time;
  using ::timespec_get _LIBCPP_USING_IF_EXISTS;
} // export
