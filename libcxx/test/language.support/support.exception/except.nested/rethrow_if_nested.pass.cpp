//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <exception>

// class nested_exception;

// template <class E> void rethrow_if_nested(const E& e);

#include <exception>
#include <cstdlib>
#include <cassert>

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}
    virtual ~A() _NOEXCEPT {}

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
                catch (const B& b)
                {
                    assert(b == B(5));
                }
            }
        }
    }
    {
        try
        {
            std::rethrow_if_nested(1);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
}
