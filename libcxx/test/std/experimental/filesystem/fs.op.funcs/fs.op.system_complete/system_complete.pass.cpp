//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// path system_complete(const path& p);
// path system_complete(const path& p, error_code& ec);

// Note: For POSIX based operating systems, 'system_complete(p)' has the
// same semantics as 'absolute(p, current_path())'.

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

TEST_SUITE(filesystem_system_complete_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(system_complete(p));
    ASSERT_NOT_NOEXCEPT(system_complete(p, ec));
}


TEST_CASE(basic_system_complete_tests)
{
    const path testCases[] = {
        "//net/foo", //  has_root_name() &&  has_root_directory()
        "/foo",      // !has_root_name() &&  has_root_directory()
        "//net",     //  has_root_name() && !has_root_directory()
        "bar/baz"    // !has_root_name() && !has_root_directory()
    };
    const path base = current_path();
    for (auto& p : testCases) {
        const path ret = system_complete(p);
        const path expect = absolute(p, base);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret == expect);
    }
}

TEST_SUITE_END()
