// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <__config>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#  include <ostream>
#endif

export module std:ostream;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::basic_ostream;

  using std::ostream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wostream;
#  endif

  using std::endl;
  using std::ends;
  using std::flush;

#  if 0
  using std::emit_on_flush;
  using std::flush_emit;
  using std::noemit_on_flush;
#  endif
  using std::operator<<;

#  if 0
  // [ostream.formatted.print], print functions
  using std::print;
  using std::println;

  using std::vprint_nonunicode;
  using std::vprint_unicode;
#  endif
} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
