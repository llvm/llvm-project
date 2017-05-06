//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MSVC_STDLIB_FORCE_INCLUDE_HPP
#define SUPPORT_MSVC_STDLIB_FORCE_INCLUDE_HPP

// This header is force-included when running the libc++ tests against the
// MSVC standard library.

// Silence warnings about CRT machinery.
#define _CRT_SECURE_NO_WARNINGS

// Avoid assertion dialogs.
#define _CRT_SECURE_INVALID_PARAMETER(EXPR) ::abort()

#include <crtdbg.h>
#include <stdlib.h>

#if defined(_LIBCPP_VERSION)
    #error This header may not be used when targeting libc++
#endif

// Indicates that we are using the MSVC standard library.
#ifndef _MSVC_STL_VER
    #define _MSVC_STL_VER 42
#endif

struct AssertionDialogAvoider {
    AssertionDialogAvoider() {
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);

        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    }
};

const AssertionDialogAvoider assertion_dialog_avoider{};

// MSVC frontend only configurations
#if !defined(__clang__)
    // Simulate feature-test macros.
    #define __has_feature(X) _MSVC_HAS_FEATURE_ ## X
    #define _MSVC_HAS_FEATURE_cxx_exceptions    1
    #define _MSVC_HAS_FEATURE_cxx_rtti          1
    #define _MSVC_HAS_FEATURE_address_sanitizer 0
    #define _MSVC_HAS_FEATURE_memory_sanitizer  0
    #define _MSVC_HAS_FEATURE_thread_sanitizer  0

    // Silence compiler warnings.
    #pragma warning(disable: 4180) // qualifier applied to function type has no meaning; ignored
    #pragma warning(disable: 4521) // multiple copy constructors specified
    #pragma warning(disable: 4702) // unreachable code
    #pragma warning(disable: 28251) // Inconsistent annotation for 'new': this instance has no annotations.
#endif // !defined(__clang__)

// MSVC doesn't have __int128_t.
#define _LIBCPP_HAS_NO_INT128

// MSVC has quick_exit() and at_quick_exit().
#define _LIBCPP_HAS_QUICK_EXIT

// atomic_is_lock_free.pass.cpp needs this VS 2015 Update 2 fix.
#define _ENABLE_ATOMIC_ALIGNMENT_FIX

// Enable features that /std:c++latest removes by default.
#define _HAS_AUTO_PTR_ETC          1
#define _HAS_FUNCTION_ASSIGN       1
#define _HAS_OLD_IOSTREAMS_MEMBERS 1

// Silence warnings about raw pointers and other unchecked iterators.
#define _SCL_SECURE_NO_WARNINGS

#include <ciso646>

#if _HAS_CXX17
    #define TEST_STD_VER 17
#else // _HAS_CXX17
    #define TEST_STD_VER 14
#endif // _HAS_CXX17

#endif // SUPPORT_MSVC_STDLIB_FORCE_INCLUDE_HPP
