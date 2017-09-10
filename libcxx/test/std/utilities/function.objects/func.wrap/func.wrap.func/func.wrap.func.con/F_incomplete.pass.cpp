//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// template<class F> function(F);

// Allow incomplete argument types in the __is_callable check

#include <functional>

struct X{
    typedef std::function<void(X&)> callback_type;
    virtual ~X() {}
private:
    callback_type _cb;
};

int main()
{
}
