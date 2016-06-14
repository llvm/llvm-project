//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T... I>
// struct integer_sequence
// {
//     typedef T type;
//
//     static constexpr size_t size() noexcept;
// };

// This test is a conforming extension.  The extension turns undefined behavior
//  into a compile-time error.

#include <utility>

int main()
{
#if _LIBCPP_STD_VER > 11

//  Should fail to compile, since float is not an integral type
    using floatmix = std::integer_sequence<float>;
    floatmix::value_type I;

#else

X

#endif  // _LIBCPP_STD_VER > 11
}
