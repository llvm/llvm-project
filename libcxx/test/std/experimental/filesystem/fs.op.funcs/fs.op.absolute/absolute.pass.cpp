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

// path absolute(const path& p, const path& base=current_path());

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

TEST_SUITE(filesystem_absolute_path_test_suite)

TEST_CASE(absolute_signature_test)
{
    const path p; ((void)p);
    ASSERT_NOT_NOEXCEPT(absolute(p));
    ASSERT_NOT_NOEXCEPT(absolute(p, p));
}

// There are 4 cases is the proposal for absolute path.
// Each scope tests one of the cases.
TEST_CASE(absolute_path_test)
{
    // has_root_name() && has_root_directory()
    {
        const path p("//net/foo");
        const path base("//net/bar/baz");
        TEST_REQUIRE(p.has_root_name());
        TEST_REQUIRE(p.has_root_directory());
        TEST_CHECK(p.is_absolute());
        path ret = absolute(p, base);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret == p);
    }
    // !has_root_name() && has_root_directory()
    {
        const path p("/foo");
        const path base("//net/bar");
        TEST_REQUIRE(not p.has_root_name());
        TEST_REQUIRE(p.has_root_directory());
        TEST_CHECK(p.is_absolute());
        // ensure absolute(base) is not recursively called
        TEST_REQUIRE(base.has_root_name());
        TEST_REQUIRE(base.has_root_directory());

        path ret = absolute(p, base);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret.has_root_name());
        TEST_CHECK(ret.root_name() == path("//net"));
        TEST_CHECK(ret.has_root_directory());
        TEST_CHECK(ret.root_directory() == path("/"));
        TEST_CHECK(ret == path("//net/foo"));
    }
    // has_root_name() && !has_root_directory()
    {
        const path p("//net");
        const path base("//net/foo/bar");
        TEST_REQUIRE(p.has_root_name());
        TEST_REQUIRE(not p.has_root_directory());
        TEST_CHECK(not p.is_absolute());
        // absolute is called recursively on base. The following conditions
        // must be true for it to return base unmodified
        TEST_REQUIRE(base.has_root_name());
        TEST_REQUIRE(base.has_root_directory());
        path ret = absolute(p, base);
        const path expect("//net/foo/bar");
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret == path("//net/foo/bar"));
    }
    // !has_root_name() && !has_root_directory()
    {
        const path p("bar/baz");
        const path base("//net/foo");
        TEST_REQUIRE(not p.has_root_name());
        TEST_REQUIRE(not p.has_root_directory());
        TEST_REQUIRE(base.has_root_name());
        TEST_REQUIRE(base.has_root_directory());

        path ret = absolute(p, base);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret == path("//net/foo/bar/baz"));
    }
}

TEST_CASE(absolute_path_with_default_base)
{
    const path testCases[] = {
        "//net/foo", //  has_root_name() &&  has_root_directory()
        "/foo",      // !has_root_name() &&  has_root_directory()
        "//net",     //  has_root_name() && !has_root_directory()
        "bar/baz"    // !has_root_name() && !has_root_directory()
    };
    const path base = current_path();
    for (auto& p : testCases) {
        const path ret = absolute(p);
        const path expect = absolute(p, base);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(ret == expect);
    }
}

TEST_SUITE_END()
