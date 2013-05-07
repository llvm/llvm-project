//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// pointer operator->() const;

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198

#include <iterator>
#include <cassert>

class A
{
    int data_;
public:
    A() : data_(1) {}
    ~A() {data_ = -1;}

    int get() const {return data_;}

    friend bool operator==(const A& x, const A& y)
        {return x.data_ == y.data_;}
};

template <class It>
class weird_iterator
{
    It it_;
public:
    typedef It                              value_type;
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef std::ptrdiff_t                  difference_type;
    typedef It*                             pointer;
    typedef It&                             reference;

    weird_iterator() {}
    explicit weird_iterator(It it) : it_(it) {}
    ~weird_iterator() {it_ = It();}

    reference operator*() {return it_;}
    pointer operator->() {return &it_;}

    weird_iterator& operator--() {return *this;}
};

template <class It>
void
test(It i, typename std::iterator_traits<It>::value_type x)
{
    std::reverse_iterator<It> r(i);
    assert(r->get() == x.get());
}

int main()
{
    test(weird_iterator<A>(A()), A());
    A a;
    test(&a+1, A());
}
