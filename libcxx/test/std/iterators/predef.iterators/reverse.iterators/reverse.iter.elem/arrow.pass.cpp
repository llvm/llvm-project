//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// pointer operator->() const; // constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360


#include <iterator>
#include <list>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 20
// C++20 bidirectional_iterator that does not satisfy the Cpp17BidirectionalIterator named requirement.
template <class It>
class cpp20_bidirectional_iterator_with_arrow {
  It it_;

public:
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = std::bidirectional_iterator_tag;
  using value_type        = std::iterator_traits<It>::value_type;
  using difference_type   = std::iterator_traits<It>::difference_type;

  cpp20_bidirectional_iterator_with_arrow() : it_() {}
  explicit cpp20_bidirectional_iterator_with_arrow(It it) : it_(it) {}

  decltype(auto) operator*() const { return *it_; }

  auto operator->() const {
    if constexpr (std::is_pointer_v<It>) {
      return it_;
    } else {
      return it_.operator->();
    }
  }

  cpp20_bidirectional_iterator_with_arrow& operator++() {
    ++it_;
    return *this;
  }
  cpp20_bidirectional_iterator_with_arrow& operator--() {
    --it_;
    return *this;
  }
  cpp20_bidirectional_iterator_with_arrow operator++(int) { return cpp20_bidirectional_iterator_with_arrow(it_++); }
  cpp20_bidirectional_iterator_with_arrow operator--(int) { return cpp20_bidirectional_iterator_with_arrow(it_--); }

  friend bool
  operator==(const cpp20_bidirectional_iterator_with_arrow& x, const cpp20_bidirectional_iterator_with_arrow& y) {
    return x.it_ == y.it_;
  }
  friend bool
  operator!=(const cpp20_bidirectional_iterator_with_arrow& x, const cpp20_bidirectional_iterator_with_arrow& y) {
    return x.it_ != y.it_;
  }

  friend It base(const cpp20_bidirectional_iterator_with_arrow& i) { return i.it_; }
};
#endif

class A
{
    int data_;
public:
    A() : data_(1) {}
    A(const A&) = default;
    A& operator=(const A&) = default;
    ~A() {data_ = -1;}

    int get() const {return data_;}

    friend bool operator==(const A& x, const A& y)
        {return x.data_ == y.data_;}
};

template <class It>
void
test(It i, typename std::iterator_traits<It>::value_type x)
{
    std::reverse_iterator<It> r(i);
    assert(r->get() == x.get());
}

class B
{
    int data_;
public:
    B(int d=1) : data_(d) {}
    B(const B&) = default;
    B& operator=(const B&) = default;
    ~B() {data_ = -1;}

    int get() const {return data_;}

    friend bool operator==(const B& x, const B& y)
        {return x.data_ == y.data_;}
    const B *operator&() const { return nullptr; }
    B       *operator&()       { return nullptr; }
};

class C
{
    int data_;
public:
    TEST_CONSTEXPR C() : data_(1) {}

    TEST_CONSTEXPR int get() const {return data_;}

    friend TEST_CONSTEXPR bool operator==(const C& x, const C& y)
        {return x.data_ == y.data_;}
};

TEST_CONSTEXPR  C gC;

int main(int, char**)
{
  A a;
  test(&a+1, A());

  {
    std::list<B> l;
    l.push_back(B(0));
    l.push_back(B(1));
    l.push_back(B(2));

    {
      std::list<B>::const_iterator i = l.begin();
      assert ( i->get() == 0 );  ++i;
      assert ( i->get() == 1 );  ++i;
      assert ( i->get() == 2 );  ++i;
      assert ( i == l.end ());
    }

    {
      std::list<B>::const_reverse_iterator ri = l.rbegin();
      assert ( ri->get() == 2 );  ++ri;
      assert ( ri->get() == 1 );  ++ri;
      assert ( ri->get() == 0 );  ++ri;
      assert ( ri == l.rend ());
    }
  }

#if TEST_STD_VER > 14
  {
    typedef std::reverse_iterator<const C *> RI;
    constexpr RI it1 = std::make_reverse_iterator(&gC+1);

    static_assert(it1->get() == gC.get(), "");
  }
#endif
#if TEST_STD_VER >= 20
  {
    // The underlying iterator models c++20 bidirectional_iterator,
    // but does not satisfy c++17 BidirectionalIterator named requirement
    B data[] = {1, 2, 3};
    cpp20_bidirectional_iterator_with_arrow<B*> iter(data + 3);
    auto ri = std::make_reverse_iterator(iter);
    assert(ri->get() == 3);
  }
#endif
  {
    ((void)gC);
  }

  return 0;
}
