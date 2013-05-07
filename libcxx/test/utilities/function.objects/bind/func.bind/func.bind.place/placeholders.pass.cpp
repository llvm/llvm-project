//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// placeholders

#include <functional>

template <class T>
void
test(const T& t)
{
    T t2;
    T t3 = t;
}

int main()
{
    test(std::placeholders::_1);
    test(std::placeholders::_2);
    test(std::placeholders::_3);
    test(std::placeholders::_4);
    test(std::placeholders::_5);
    test(std::placeholders::_6);
    test(std::placeholders::_7);
    test(std::placeholders::_8);
    test(std::placeholders::_9);
    test(std::placeholders::_10);
}
