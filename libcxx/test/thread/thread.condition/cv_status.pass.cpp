//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// enum class cv_status { no_timeout, timeout };

#include <condition_variable>
#include <cassert>

int main()
{
    assert(std::cv_status::no_timeout == 0);
    assert(std::cv_status::timeout == 1);
}
