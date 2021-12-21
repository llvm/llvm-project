//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANDOM_RANDOM_DEVICE_H
#define _LIBCPP___RANDOM_RANDOM_DEVICE_H

#include <__config>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANDOM_DEVICE)

// Libc++ supports various implementations of std::random_device.
//
// _LIBCPP_USING_DEV_RANDOM
//      Read entropy from the given file, by default `/dev/urandom`.
//      If a token is provided, it is assumed to be the path to a file
//      to read entropy from. This is the default behavior if nothing
//      else is specified. This implementation requires storing state
//      inside `std::random_device`.
//
// _LIBCPP_USING_ARC4_RANDOM
//      Use arc4random(). This allows obtaining random data even when
//      using sandboxing mechanisms. On some platforms like Apple, this
//      is the recommended source of entropy for user-space programs.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_GETENTROPY
//      Use getentropy().
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_NACL_RANDOM
//      NaCl's sandbox (which PNaCl also runs in) doesn't allow filesystem access,
//      including accesses to the special files under `/dev`. This implementation
//      uses the NaCL syscall `nacl_secure_random_init()` to get entropy.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_WIN32_RANDOM
//      Use rand_s(), for use on Windows.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
#if defined(__OpenBSD__)
#  define _LIBCPP_USING_ARC4_RANDOM
#elif defined(__Fuchsia__) || defined(__wasi__)
#  define _LIBCPP_USING_GETENTROPY
#elif defined(__native_client__)
#  define _LIBCPP_USING_NACL_RANDOM
#elif defined(_LIBCPP_WIN32API)
#  define _LIBCPP_USING_WIN32_RANDOM
#else
#  define _LIBCPP_USING_DEV_RANDOM
#endif

class _LIBCPP_TYPE_VIS random_device
{
#ifdef _LIBCPP_USING_DEV_RANDOM
    int __f_;
#endif
public:
    // types
    typedef unsigned result_type;

    // generator characteristics
    static _LIBCPP_CONSTEXPR const result_type _Min = 0;
    static _LIBCPP_CONSTEXPR const result_type _Max = 0xFFFFFFFFu;

    _LIBCPP_INLINE_VISIBILITY
    static _LIBCPP_CONSTEXPR result_type min() { return _Min;}
    _LIBCPP_INLINE_VISIBILITY
    static _LIBCPP_CONSTEXPR result_type max() { return _Max;}

    // constructors
#ifndef _LIBCPP_CXX03_LANG
    random_device() : random_device("/dev/urandom") {}
    explicit random_device(const string& __token);
#else
    explicit random_device(const string& __token = "/dev/urandom");
#endif
    ~random_device();

    // generating functions
    result_type operator()();

    // property functions
    double entropy() const _NOEXCEPT;

    random_device(const random_device&) = delete;
    void operator=(const random_device&) = delete;
};

#endif // !_LIBCPP_HAS_NO_RANDOM_DEVICE

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANDOM_RANDOM_DEVICE_H
