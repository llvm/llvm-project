// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
  using ::lconv;

  using ::localeconv;
  using ::setlocale;
#endif // _LIBCPP_HAS_NO_LOCALIZATION
} // export
