//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(_LIBCPP_ABI_DO_NOT_RETAIN_SHRINKING_RESERVE)

// Instantiate a copy of the shrinking reserve implementation to maintain ABI compatibility for older versions of
// basic_string which relied on this behavior in move assignment.
#  define _LIBCPP_ENABLE_RESERVE_SHRINKING_ABI
#  include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

template _LIBCPP_EXPORTED_FROM_ABI void basic_string<char>::reserve(size_type);
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template _LIBCPP_EXPORTED_FROM_ABI void basic_string<wchar_t >::reserve(size_type);
#  endif

_LIBCPP_END_NAMESPACE_STD

#endif
