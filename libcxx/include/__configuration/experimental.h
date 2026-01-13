//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_EXPERIMENTAL_H
#define _LIBCPP___CONFIGURATION_EXPERIMENTAL_H

#include <__config_site>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#if __has_feature(experimental_library)
#  ifndef _LIBCPP_ENABLE_EXPERIMENTAL
#    define _LIBCPP_ENABLE_EXPERIMENTAL
#  endif
#endif

// Incomplete features get their own specific disabling flags. This makes it
// easier to grep for target specific flags once the feature is complete.
#if defined(_LIBCPP_ENABLE_EXPERIMENTAL) || defined(_LIBCPP_BUILDING_LIBRARY)
#  define _LIBCPP_HAS_EXPERIMENTAL_LIBRARY 1
#else
#  define _LIBCPP_HAS_EXPERIMENTAL_LIBRARY 0
#endif

#define _LIBCPP_HAS_EXPERIMENTAL_PSTL _LIBCPP_HAS_EXPERIMENTAL_LIBRARY
#define _LIBCPP_HAS_EXPERIMENTAL_TZDB _LIBCPP_HAS_EXPERIMENTAL_LIBRARY
#define _LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM _LIBCPP_HAS_EXPERIMENTAL_LIBRARY
#define _LIBCPP_HAS_EXPERIMENTAL_HARDENING_OBSERVE_SEMANTIC _LIBCPP_HAS_EXPERIMENTAL_LIBRARY
#define _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR _LIBCPP_HAS_EXPERIMENTAL_LIBRARY

#endif // _LIBCPP___CONFIGURATION_EXPERIMENTAL_H
