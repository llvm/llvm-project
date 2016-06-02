//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// Example move-only deleter

#ifndef DELETER_H
#define DELETER_H

#include <type_traits>
#include <cassert>

#ifndef _LIBCPP_HAS_NO_DELETED_FUNCTIONS
#define DELETE_FUNCTION = delete
#else
#define DELETE_FUNCTION { assert(false); }
#endif

struct test_deleter_base
{
    static int count;
    static int dealloc_count;
};

int test_deleter_base::count = 0;
int test_deleter_base::dealloc_count = 0;

template <class T>
class test_deleter
    : public test_deleter_base
{
    int state_;

public:

    test_deleter() : state_(0) {++count;}
    explicit test_deleter(int s) : state_(s) {++count;}
    test_deleter(const test_deleter& d)
        : state_(d.state_) {++count;}
    ~test_deleter() {assert(state_ >= 0); --count; state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {assert(state_ >= 0); ++dealloc_count; delete p;}

    test_deleter* operator&() const DELETE_FUNCTION;
};

template <class T>
void
swap(test_deleter<T>& x, test_deleter<T>& y)
{
    test_deleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}

#endif  // DELETER_H
