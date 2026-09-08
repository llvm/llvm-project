//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) && !defined(__WINE__)) || defined(__CYGWIN__)
#if defined(__GNUC__) || defined(__clang__)
#if defined(_HERBCEPTIONS_BUILDING_RUNTIME)
#if 0
//clang bug on windows on arm. Wait for upstream to fix before we set weak
#define __HERBCEPTIONS_API [[__gnu__::__dllexport__, __gnu__::__weak__]]
#else
#define __HERBCEPTIONS_API [[__gnu__::__dllexport__]]
#endif
#else
#define __HERBCEPTIONS_API [[__gnu__::__dllimport__, __gnu__::__const__]]
#endif
#elif defined(_MSC_VER)
#if defined(_HERBCEPTIONS_BUILDING_RUNTIME)
#define __HERBCEPTIONS_API __declspec(dllexport)
#else
#define __HERBCEPTIONS_API __declspec(dllimport)
#endif
#endif
#else
#if defined(_HERBCEPTIONS_BUILD_RUNTIME)
#define __HERBCEPTIONS_API                                                     \
  [[__gnu__::__visibility__("default"), __gnu__::__weak__]]
#else
#define __HERBCEPTIONS_API [[__gnu__::__const__]]
#endif
#endif
