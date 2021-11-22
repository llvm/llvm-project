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
//      void __cxa_rethrow_primary_exception(void* thrown_object);
// }

unsigned gCounter = 0;

void my_terminate() { exit(0); }

int main ()
{
    // should not call std::terminate()
    __cxxabiv1::__cxa_rethrow_primary_exception(nullptr);

    std::set_terminate(my_terminate);

    // should call std::terminate()
    __cxxabiv1::__cxa_rethrow_primary_exception((void*) &gCounter);
    assert(false);

    return 0;
}
