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

// bool equivalent(path const& lhs, path const& rhs);
// bool equivalent(path const& lhs, path const& rhs, std::error_code& ec) noexcept;

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

TEST_SUITE(equivalent_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOEXCEPT(equivalent(p, p, ec));
    ASSERT_NOT_NOEXCEPT(equivalent(p, p));
}

TEST_CASE(equivalent_test)
{
    struct TestCase {
        path lhs;
        path rhs;
        bool expect;
    };
    const TestCase testCases[] = {
        {StaticEnv::Dir, StaticEnv::Dir, true},
        {StaticEnv::File, StaticEnv::Dir, false},
        {StaticEnv::Dir, StaticEnv::SymlinkToDir, true},
        {StaticEnv::Dir, StaticEnv::SymlinkToFile, false},
        {StaticEnv::File, StaticEnv::File, true},
        {StaticEnv::File, StaticEnv::SymlinkToFile, true},
    };
    for (auto& TC : testCases) {
        std::error_code ec;
        TEST_CHECK(equivalent(TC.lhs, TC.rhs, ec) == TC.expect);
        TEST_CHECK(!ec);
    }
}

TEST_CASE(equivalent_reports_double_dne)
{
    const path E = StaticEnv::File;
    const path DNE = StaticEnv::DNE;
    { // Test that no exception is thrown if one of the paths exists
        TEST_CHECK(equivalent(E, DNE) == false);
        TEST_CHECK(equivalent(DNE, E) == false);
    }
    { // Test that an exception is thrown if both paths do not exist.
        TEST_CHECK_THROW(filesystem_error, equivalent(DNE, DNE));
    }
    {
        std::error_code ec;
        TEST_CHECK(equivalent(DNE, DNE, ec) == false);
        TEST_CHECK(ec);
    }
}

TEST_CASE(equivalent_is_other_succeeds)
{
    scoped_test_env env;
    path const file = env.create_file("file", 42);
    const path hl1 = env.create_hardlink(file, "hl1");
    const path hl2 = env.create_hardlink(file, "hl2");
    TEST_CHECK(equivalent(file, hl1));
    TEST_CHECK(equivalent(file, hl2));
    TEST_CHECK(equivalent(hl1, hl2));
}

TEST_SUITE_END()
