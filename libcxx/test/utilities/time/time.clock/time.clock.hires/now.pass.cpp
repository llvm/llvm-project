//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// high_resolution_clock

// static time_point now();

#include <chrono>

int main()
{
    typedef std::chrono::high_resolution_clock C;
    C::time_point t1 = C::now();
}
