//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <exception>

// class nested_exception;

// nested_exception& operator=(const nested_exception&) throw() = default;

#include <exception>
#include <cassert>

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

int main()
{
    {
        std::nested_exception e0;
        std::nested_exception e;
        e = e0;
        assert(e.nested_ptr() == nullptr);
    }
    {
        try
        {
            throw A(2);
            assert(false);
        }
        catch (const A&)
        {
            std::nested_exception e0;
            std::nested_exception e;
            e = e0;
            assert(e.nested_ptr() != nullptr);
            try
            {
                rethrow_exception(e.nested_ptr());
                assert(false);
            }
            catch (const A& a)
            {
                assert(a == A(2));
            }
        }
    }
}
