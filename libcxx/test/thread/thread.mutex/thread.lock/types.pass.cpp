//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// struct defer_lock_t {};
// struct try_to_lock_t {};
// struct adopt_lock_t {};
//
// constexpr defer_lock_t  defer_lock{};
// constexpr try_to_lock_t try_to_lock{};
// constexpr adopt_lock_t  adopt_lock{};

#include <mutex>
#include <type_traits>

int main()
{
    typedef std::defer_lock_t T1;
    typedef std::try_to_lock_t T2;
    typedef std::adopt_lock_t T3;

    T1 t1 = std::defer_lock;
    T2 t2 = std::try_to_lock;
    T3 t3 = std::adopt_lock;
}
