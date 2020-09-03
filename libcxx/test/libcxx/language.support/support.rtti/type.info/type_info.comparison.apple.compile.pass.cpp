//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test makes sure that we do assume that type_infos are unique across
// all translation units on Apple platforms. See https://llvm.org/PR45549.

// TODO:
// We don't really want to require 'darwin' here -- instead we'd like to express
// that this test requires the flavor of libc++ built by Apple, which we don't
// have a clear way to express right now. If another flavor of libc++ was built
// targetting Apple platforms without assuming merged RTTI, this test would fail.
// REQUIRES: darwin

#include <typeinfo>

#if !defined(_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION)
#   error "_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION should be defined on Apple platforms"
#endif

#if _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION != 1
#   error "_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION should be 1 (assume RTTI is merged) on Apple platforms"
#endif
