//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T>
//   requires MoveAssignable<T> && MoveConstructible<T>
//   void
//   swap(T& a, T& b);

#include <utility>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
#include <memory>
#endif

void
test()
{
    int i = 1;
    int j = 2;
    std::swap(i, j);
    assert(i == 2);
    assert(j == 1);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

void
test1()
{
    std::unique_ptr<int> i(new int(1));
    std::unique_ptr<int> j(new int(2));
    std::swap(i, j);
    assert(*i == 2);
    assert(*j == 1);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    test();
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test1();
#endif
}
