//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <iterator>

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

#include <list>
#include <memory>
#include <cassert>

template <class C>
void
test(C c)
{
    std::front_insert_iterator<C> i(c);
    i = typename C::value_type();
    assert(c.front() == typename C::value_type());
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test(std::list<std::unique_ptr<int> >());
#endif
}
