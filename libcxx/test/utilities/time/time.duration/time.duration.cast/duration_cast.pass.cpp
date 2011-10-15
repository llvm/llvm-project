//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class ToDuration, class Rep, class Period>
//   ToDuration
//   duration_cast(const duration<Rep, Period>& d);

#include <chrono>
#include <type_traits>
#include <cassert>

template <class ToDuration, class FromDuration>
void
test(const FromDuration& f, const ToDuration& d)
{
    typedef decltype(std::chrono::duration_cast<ToDuration>(f)) R;
    static_assert((std::is_same<R, ToDuration>::value), "");
    assert(std::chrono::duration_cast<ToDuration>(f) == d);
}

int main()
{
    test(std::chrono::milliseconds(7265000), std::chrono::hours(2));
    test(std::chrono::milliseconds(7265000), std::chrono::minutes(121));
    test(std::chrono::milliseconds(7265000), std::chrono::seconds(7265));
    test(std::chrono::milliseconds(7265000), std::chrono::milliseconds(7265000));
    test(std::chrono::milliseconds(7265000), std::chrono::microseconds(7265000000LL));
    test(std::chrono::milliseconds(7265000), std::chrono::nanoseconds(7265000000000LL));
    test(std::chrono::milliseconds(7265000),
         std::chrono::duration<double, std::ratio<3600> >(7265./3600));
    test(std::chrono::duration<int, std::ratio<2, 3> >(9),
         std::chrono::duration<int, std::ratio<3, 5> >(10));
}
