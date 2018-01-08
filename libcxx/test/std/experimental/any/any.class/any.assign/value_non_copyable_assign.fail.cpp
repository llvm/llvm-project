//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// template <class Value>
// any& operator=(Value &&);

// Instantiate the value assignment operator with a non-copyable type.

#include <experimental/any>

class non_copyable
{
    non_copyable(non_copyable const &);

public:
    non_copyable() {}
    non_copyable(non_copyable &&) {}
};

int main()
{
    using namespace std::experimental;
    non_copyable nc;
    any a;
    a = static_cast<non_copyable &&>(nc); // expected-error-re@experimental/any:* 2 {{static_assert failed{{.*}} "_ValueType must be CopyConstructible."}}
        // expected-error@experimental/any:* {{calling a private constructor of class 'non_copyable'}}

}
