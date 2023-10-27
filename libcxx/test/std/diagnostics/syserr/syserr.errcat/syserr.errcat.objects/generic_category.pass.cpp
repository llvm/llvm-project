//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: suse-linux-enterprise-server-11
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <system_error>

// class error_category

// const error_category& generic_category();

#include <system_error>
#include <cassert>
#include <string>
#include <cerrno>

#include "test_macros.h"

// See https://llvm.org/D65667
struct StaticInit {
    const std::error_category* ec;
    ~StaticInit() {
        std::string str = ec->name();
        assert(str == "generic") ;
    }
};
static StaticInit foo;

int main(int, char**)
{
    {
        const std::error_category& e_cat1 = std::generic_category();
        std::string m1 = e_cat1.name();
        assert(m1 == "generic");
    }

    // Test the result of message(int cond) when given a bad error condition
    {
        errno = E2BIG; // something that message will never generate
        const std::error_category& e_cat1 = std::generic_category();
        const std::string msg = e_cat1.message(-1);
        // Exact message format varies by platform.
#if defined(_AIX)
        LIBCPP_ASSERT(msg.rfind("Error -1 occurred", 0) == 0);
#else
        LIBCPP_ASSERT(msg.rfind("Unknown error", 0) == 0);
#endif
        assert(errno == E2BIG);
    }

    {
        foo.ec = &std::generic_category();
        std::string m2 = foo.ec->name();
        assert(m2 == "generic");
    }

    return 0;
}
