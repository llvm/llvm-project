// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_NAMESPACE_H
#define _LIBCPP___CONFIGURATION_NAMESPACE_H

#include <__config_site>
#include <__configuration/attributes.h>
#include <__configuration/diagnostic_suppression.h>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

// Clang modules take a significant compile time hit when pushing and popping diagnostics.
// Since all the headers are marked as system headers unless _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER is defined, we can
// simply disable this pushing and popping when _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER isn't defined.
#ifdef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  define _LIBCPP_PUSH_EXTENSION_DIAGNOSTICS                                                                           \
    _LIBCPP_DIAGNOSTIC_PUSH                                                                                            \
    _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc++11-extensions")                                                             \
    _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc++14-extensions")                                                             \
    _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc++17-extensions")                                                             \
    _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc++20-extensions")                                                             \
    _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc++23-extensions")                                                             \
    _LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wc++14-extensions")                                                               \
    _LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wc++17-extensions")                                                               \
    _LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wc++20-extensions")                                                               \
    _LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wc++23-extensions")
#  define _LIBCPP_POP_EXTENSION_DIAGNOSTICS _LIBCPP_DIAGNOSTIC_POP
#else
#  define _LIBCPP_PUSH_EXTENSION_DIAGNOSTICS
#  define _LIBCPP_POP_EXTENSION_DIAGNOSTICS
#endif

// clang-format off

// The unversioned namespace is used when we want to be ABI compatible with other standard libraries in some way. There
// are two main categories where that's the case:
// - Historically, we have made exception types ABI compatible with libstdc++ to allow throwing them between libstdc++
//   and libc++. This is not used anymore for new exception types, since there is no use-case for it anymore.
// - Types and functions which are used by the compiler are in the unversioned namespace, since the compiler has to know
//   their mangling without the appropriate declaration in some cases.
// If it's not clear whether using the unversioned namespace is the correct thing to do, it's not. The versioned
// namespace (_LIBCPP_BEGIN_NAMESPACE_STD) should almost always be used.
#  define _LIBCPP_BEGIN_UNVERSIONED_NAMESPACE_STD                                                                      \
    _LIBCPP_PUSH_EXTENSION_DIAGNOSTICS namespace _LIBCPP_NAMESPACE_VISIBILITY std {

#  define _LIBCPP_END_UNVERSIONED_NAMESPACE_STD } _LIBCPP_POP_EXTENSION_DIAGNOSTICS

#  define _LIBCPP_BEGIN_NAMESPACE_STD _LIBCPP_BEGIN_UNVERSIONED_NAMESPACE_STD inline namespace _LIBCPP_ABI_NAMESPACE {
#  define _LIBCPP_END_NAMESPACE_STD } _LIBCPP_END_UNVERSIONED_NAMESPACE_STD

// TODO: This should really be in the versioned namespace
#define _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL _LIBCPP_BEGIN_UNVERSIONED_NAMESPACE_STD namespace experimental {
#define _LIBCPP_END_NAMESPACE_EXPERIMENTAL } _LIBCPP_END_UNVERSIONED_NAMESPACE_STD

#define _LIBCPP_BEGIN_NAMESPACE_LFTS _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL inline namespace fundamentals_v1 {
#define _LIBCPP_END_NAMESPACE_LFTS } _LIBCPP_END_NAMESPACE_EXPERIMENTAL

#define _LIBCPP_BEGIN_NAMESPACE_LFTS_V2 _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL inline namespace fundamentals_v2 {
#define _LIBCPP_END_NAMESPACE_LFTS_V2 } _LIBCPP_END_NAMESPACE_EXPERIMENTAL

#ifdef _LIBCPP_ABI_NO_FILESYSTEM_INLINE_NAMESPACE
#  define _LIBCPP_BEGIN_NAMESPACE_FILESYSTEM _LIBCPP_BEGIN_NAMESPACE_STD namespace filesystem {
#  define _LIBCPP_END_NAMESPACE_FILESYSTEM } _LIBCPP_END_NAMESPACE_STD
#else
#  define _LIBCPP_BEGIN_NAMESPACE_FILESYSTEM _LIBCPP_BEGIN_NAMESPACE_STD                                               \
                                             inline namespace __fs { namespace filesystem {

#  define _LIBCPP_END_NAMESPACE_FILESYSTEM }} _LIBCPP_END_NAMESPACE_STD
#endif

// clang-format on

#endif // _LIBCPP___CONFIGURATION_NAMESPACE_H
