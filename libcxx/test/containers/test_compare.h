#ifndef TEST_COMPARE_H
#define TEST_COMPARE_H

#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <new>
#include <climits>

template <class C>
class test_compare
    : private C
{
    int data_;
public:
    explicit test_compare(int data = 0) : data_(data) {}

    typename C::result_type
    operator()(typename std::add_lvalue_reference<const typename C::first_argument_type>::type x,
               typename std::add_lvalue_reference<const typename C::second_argument_type>::type y) const
        {return C::operator()(x, y);}

    bool operator==(const test_compare& c) const
        {return data_ == c.data_;}
};

#endif  // TEST_COMPARE_H
