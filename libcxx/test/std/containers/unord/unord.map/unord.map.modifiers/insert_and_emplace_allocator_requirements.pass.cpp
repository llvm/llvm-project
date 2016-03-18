//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_map

// insert(...);

// UNSUPPORTED: c++98, c++03


#include <unordered_map>
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
  typedef std::pair<typename C::iterator, bool> R;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    PRINT("Testing C::insert(const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.insert(v2).second == false);
    }
  }
  {
    PRINT("Testing C::insert(value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.insert(v2).second == false);
    }
  }
  {
    PRINT("Testing C::insert(value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    assert(c.insert(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.insert(std::move(v2)).second == false);
    }
  }
  {
    PRINT("Testing C::insert(const value_type&&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.insert(std::move(v2)).second == false);
    }
  }
  {
    PRINT("Testing C::insert(std::initializer_list<ValueTp>)");
    Container c;
    std::initializer_list<ValueTp> il = { ValueTp(1, 1), ValueTp(2, 1) };
    cc->expect<ValueTp const&>(2);
    c.insert(il);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(il);
    }
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type const&");
    Container c;
    const ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1), ValueTp(3, 1) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(std::begin(ValueList), std::end(ValueList));
    }
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type&&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp&&>(3);
    c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList)),
             std::move_iterator<ValueTp*>(std::end(ValueList)));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp ValueList2[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
      c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList2)),
               std::move_iterator<ValueTp*>(std::end(ValueList2)));
    }
  }
  {
    PRINT("Testing C::insert(Iter, Iter) for *Iter = value_type&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(std::begin(ValueList), std::end(ValueList));
    }
  }
}


template <class Container>
void testContainerEmplace()
{
  typedef typename Container::value_type ValueTp;
  typedef typename Container::key_type Key;
  typedef typename Container::mapped_type Mapped;
  typedef typename std::pair<Key, Mapped> NonConstKeyPair;
  typedef Container C;
  typedef std::pair<typename C::iterator, bool> R;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    PRINT("Testing C::emplace(const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    PRINT("Testing C::emplace(value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    PRINT("Testing C::emplace(value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
  {
    PRINT("Testing C::emplace(const value_type&&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
  {
    PRINT("Testing C::emplace(pair<Key, Mapped> const&)");
    Container c;
    const NonConstKeyPair v(42, 1);
    cc->expect<const NonConstKeyPair&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const NonConstKeyPair v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    PRINT("Testing C::emplace(pair<Key, Mapped> &&)");
    Container c;
    NonConstKeyPair v(42, 1);
    cc->expect<NonConstKeyPair&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      NonConstKeyPair v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
}


int main()
{
  testContainerInsert<TCT::unordered_map<> >();
  testContainerEmplace<TCT::unordered_map<> >();
}
