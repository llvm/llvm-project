/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Declarations common across all programs for all hosts and targets.
 */

#ifndef UNIVERSAL_DEFS_H_
#define UNIVERSAL_DEFS_H_

// Only use __has_cpp_attribute in C++ mode. GCC defines __has_cpp_attribute in
// C mode, but the :: in __has_cpp_attribute(scoped::attribute) is invalid.
#ifndef FLANG_HAS_CPP_ATTRIBUTE
#if defined(__cplusplus) && defined(__has_cpp_attribute)
# define FLANG_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
# define FLANG_HAS_CPP_ATTRIBUTE(x) 0
#endif
#endif

/// FLANG_FALLTHROUGH - Mark fallthrough cases in switch statements.
#if defined(__cplusplus) && __cplusplus > 201402L && FLANG_HAS_CPP_ATTRIBUTE(fallthrough)
#define FLANG_FALLTHROUGH [[fallthrough]]
#elif FLANG_HAS_CPP_ATTRIBUTE(gnu::fallthrough)
#define FLANG_FALLTHROUGH [[gnu::fallthrough]]
#elif __has_attribute(fallthrough)
#define FLANG_FALLTHROUGH __attribute__((fallthrough))
#elif FLANG_HAS_CPP_ATTRIBUTE(clang::fallthrough)
#define FLANG_FALLTHROUGH [[clang::fallthrough]]
#else
#define FLANG_FALLTHROUGH
#endif

#ifdef __cplusplus

#ifdef SHADOW_BUILD
#define BEGIN_DECL_WITH_C_LINKAGE
#define END_DECL_WITH_C_LINKAGE
#else
#define BEGIN_DECL_WITH_C_LINKAGE extern "C" {
#define END_DECL_WITH_C_LINKAGE }
#endif

#ifndef INLINE
#define INLINE inline
#define HAVE_INLINE 1
#endif

#else // !__cplusplus

#define BEGIN_DECL_WITH_C_LINKAGE
#define END_DECL_WITH_C_LINKAGE

/* Do not define the bool type if included by Flang runtime files.
   The runtime defines its own bool type. */
#ifndef FLANG_RUNTIME_GLOBAL_DEFS_H_
/* Linux and MacOS environments provide <stdbool.h> even for C89.
   Microsoft OpenTools 10 does not, even for C99. */
#if __linux__ || __APPLE__ || __STDC_VERSION__ >= 199901L && !__PGI_TOOLS10
#include <stdbool.h>
#else
typedef char bool;
#define true 1
#define false 0
#endif
#endif

#ifndef INLINE
#if defined(__GNUC__) || (__PGIC__ + 0 > 14)
#define INLINE __inline__
#define HAVE_INLINE 1
#else
#define INLINE
#endif
#endif

#endif /* __cplusplus */

#endif /* UNIVERSAL_DEFS_H_ */
