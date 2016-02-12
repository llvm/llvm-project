//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// class unordered_multiset

// insert(...)

// UNSUPPORTED: c++98, c++03

#include <unordered_set>
#include <iostream>
#include <cassert>

#include "test_macros.h"
#include "count_new.hpp"
#include "container_test_types.h"

#if TEST_STD_VER >= 11
template <class Arg>
void PrintInfo(int line, Arg&& arg)
#else
template <class Arg>
void PrintInfo(int line, Arg arg)
#endif
{
  std::cout << "In " << __FILE__ << ":" << line << ":\n    " << arg << "\n" << std::endl;
}
#define PRINT(...) PrintInfo(__LINE__, __VA_ARGS__)

template <class Container>
void testContainerInsert()
{
  typedef typename Container::value_type ValueTp;
  typedef Container C;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    PRINT("Testing C::insert(const value_type&)");
    Container c;
    const ValueTp v(42);
    cc->expect<const ValueTp&>();
    c.insert(v);
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(value_type&)");
    Container c;
    ValueTp v(42);
    cc->expect<const ValueTp&>();
    c.insert(v);
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(value_type&&)");
    Container c;
    ValueTp v(42);
    cc->expect<ValueTp&&>();
    c.insert(std::move(v));
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(std::initializer_list<ValueTp>)");
    Container c;
    std::initializer_list<ValueTp> il = { ValueTp(1), ValueTp(2) };
    cc->expect<ValueTp const&>(2);
    c.insert(il);
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type const&");
    Container c;
    const ValueTp ValueList[] = { ValueTp(1), ValueTp(2), ValueTp(3) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type&&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1), ValueTp(2) , ValueTp(3) };
    cc->expect<ValueTp&&>(3);
    c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList)),
             std::move_iterator<ValueTp*>(std::end(ValueList)));
    assert(!cc->unchecked());
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1), ValueTp(2) , ValueTp(3) };
    cc->expect<ValueTp&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
  }
}

int main()
{
  testContainerInsert<TCT::unordered_multiset<> >();
}
