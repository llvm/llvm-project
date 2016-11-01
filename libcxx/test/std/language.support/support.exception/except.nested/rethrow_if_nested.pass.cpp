//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// <exception>

// class nested_exception;

// template <class E> void rethrow_if_nested(const E& e);

#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}
    virtual ~A() TEST_NOEXCEPT {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

class B
    : public std::nested_exception,
      public A
{
public:
    explicit B(int data) : A(data) {}
    B(const B& b) : A(b) {}
};

class C
{
public:
	virtual ~C() {}
	C * operator&() const { assert(false); } // should not be called
};

int main()
{
    {
        try
        {
            A a(3);
            std::rethrow_if_nested(a);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
    {
        try
        {
            throw B(5);
        }
        catch (const B& b)
        {
            try
            {
                throw b;
            }
            catch (const A& a)
            {
                try
                {
                    std::rethrow_if_nested(a);
                    assert(false);
                }
                catch (const B& b2)
                {
                    assert(b2 == B(5));
                }
            }
        }
    }
    {
        try
        {
            std::rethrow_if_nested(C());
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }

}
