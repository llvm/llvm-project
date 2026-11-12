//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: no-exceptions

#include <cxxabi.h>
#include <exception>
#include <cassert>
#include <stdlib.h>

// namespace __cxxabiv1 {
//      void __cxa_increment_exception_refcount(void *thrown_object) throw();
// }

unsigned gCounter = 0;

void my_terminate() { exit(0); }

int main ()
{
    // should not call std::terminate()
    __cxxabiv1::__cxa_increment_exception_refcount(nullptr);

    std::set_terminate(my_terminate);

    // should call std::terminate()
    __cxxabiv1::__cxa_increment_exception_refcount((void*) &gCounter);
    assert(false);

    return 0;
}
