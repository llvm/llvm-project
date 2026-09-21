//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fallback for targets where none of aix_impl.cpp/dl_iterate_images.cpp/dyld_images.cpp provide
// `_Images::enumerate()`, e.g. bare-metal targets with no dynamic loader (no `dlfcn.h`/
// `link.h`, etc.)
#if !defined(_WIN32) && !defined(__APPLE__) && !defined(_AIX) && !(__has_include("dlfcn.h") && __has_include("link.h"))

#  include "images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

void _Images::enumerate() {}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif
