//===-------- stl_extras.h - Useful STL related functions-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_STL_EXTRAS_H
#define ORC_RT_STL_EXTRAS_H

#include <utility>
#include <tuple>

namespace __orc_rt {

/// Substitute for std::identity.
/// Switch to std::identity once we can use c++20.
template <class Ty> struct identity {
  using is_transparent = void;
  using argument_type = Ty;

  Ty &operator()(Ty &self) const { return self; }
  const Ty &operator()(const Ty &self) const { return self; }
};

} // namespace __orc_rt

#endif // ORC_RT_STL_EXTRAS
