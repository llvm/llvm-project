//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// FIXME: When modules are enabled we can't affect the contents of <mutex>
// by defining a macro
// XFAIL: -fmodules

// <mutex>

// template <class Mutex>
// class lock_guard
// {
// public:
//     typedef Mutex mutex_type;
//     ...
// };

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>
#include <type_traits>

struct NAT {};

template <class LG>
auto test_typedef(int) -> typename LG::mutex_type;

template <class LG>
auto test_typedef(...) -> NAT;

template <class LG>
constexpr bool has_mutex_type() {
    return !std::is_same<decltype(test_typedef<LG>(0)), NAT>::value;
}

int main()
{
    {
        using T = std::lock_guard<>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = std::mutex;
        using T = std::lock_guard<M1>;
        static_assert(std::is_same<T::mutex_type, M1>::value, "");
    }
    {
        using M1 = std::recursive_mutex;
        using T = std::lock_guard<M1>;
        static_assert(std::is_same<T::mutex_type, M1>::value, "");
    }
    {
        using M1 = std::mutex;
        using M2 = std::recursive_mutex;
        using T = std::lock_guard<M1, M2>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = std::mutex;
        using M2 = std::recursive_mutex;
        using T = std::lock_guard<M1, M1, M2>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = std::mutex;
        using T = std::lock_guard<M1, M1>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = std::recursive_mutex;
        using T = std::lock_guard<M1, M1, M1>;
        static_assert(!has_mutex_type<T>(), "");
    }
}
