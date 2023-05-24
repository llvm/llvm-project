// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <cfenv>

export module std:cfenv;
export namespace std {
  // types
  using std::fenv_t;
  using std::fexcept_t;

  // functions
  using std::feclearexcept;
  using std::fegetexceptflag;
  using std::feraiseexcept;
  using std::fesetexceptflag;
  using std::fetestexcept;

  using std::fegetround;
  using std::fesetround;

  using std::fegetenv;
  using std::feholdexcept;
  using std::fesetenv;
  using std::feupdateenv;

} // namespace std
