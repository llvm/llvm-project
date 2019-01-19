//===----------------------- cxa_bad_typeid.pass.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include <cxxabi.h>
#include <cassert>
#include <stdlib.h>
#include <exception>
#include <typeinfo>
#include <string>
#include <iostream>

class Base {
  virtual void foo() {};
};

class Derived : public Base {};

std::string test_bad_typeid(Derived *p) {
    return typeid(*p).name();
}

void my_terminate() { std::cout << "A" << std::endl; exit(0); }

int main ()
{
    // swap-out the terminate handler
    void (*default_handler)() = std::get_terminate(); 
    std::set_terminate(my_terminate);

#ifndef LIBCXXABI_HAS_NO_EXCEPTIONS
    try {
#endif
        test_bad_typeid(nullptr);
        assert(false);
#ifndef LIBCXXABI_HAS_NO_EXCEPTIONS
    } catch (std::bad_typeid) {
        // success
        return 0;
    } catch (...) {
        assert(false);
    }
#endif

    // failure, restore the default terminate handler and fire
    std::set_terminate(default_handler);
    std::terminate();
}
