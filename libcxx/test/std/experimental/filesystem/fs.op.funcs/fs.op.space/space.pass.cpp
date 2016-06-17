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

// space_info space(const path& p);
// space_info space(const path& p, error_code& ec) noexcept;

#include <experimental/filesystem>
#include <sys/statvfs.h>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

bool EqualDelta(std::uintmax_t x, std::uintmax_t y, std::uintmax_t delta) {
    if (x >= y) {
        return (x - y) <= delta;
    } else {
        return (y - x) <= delta;
    }
}

TEST_SUITE(filesystem_space_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(space(p)), space_info);
    ASSERT_SAME_TYPE(decltype(space(p, ec)), space_info);
    ASSERT_NOT_NOEXCEPT(space(p));
    ASSERT_NOEXCEPT(space(p, ec));
}

TEST_CASE(test_error_reporting)
{
    auto checkThrow = [](path const& f, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            space(f);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == ""
                && err.code() == ec;
        }
#else
        return true;
#endif
    };
    const path cases[] = {
        "",
        StaticEnv::DNE,
        StaticEnv::BadSymlink
    };
    for (auto& p : cases) {
        const auto expect = static_cast<std::uintmax_t>(-1);
        std::error_code ec;
        space_info info = space(p, ec);
        TEST_CHECK(ec);
        TEST_CHECK(info.capacity == expect);
        TEST_CHECK(info.free == expect);
        TEST_CHECK(info.available == expect);
        TEST_CHECK(checkThrow(p, ec));
    }
}

TEST_CASE(basic_space_test)
{
    // All the test cases should reside on the same filesystem and therefore
    // should have the same expected result. Compute this expected result
    // one and check that it looks semi-sane.
    struct statvfs expect;
    TEST_REQUIRE(::statvfs(StaticEnv::Dir.c_str(), &expect) != -1);
    TEST_CHECK(expect.f_bavail > 0);
    TEST_CHECK(expect.f_bfree > 0);
    TEST_CHECK(expect.f_bsize > 0);
    TEST_CHECK(expect.f_blocks > 0);
    const std::uintmax_t expect_cap = expect.f_blocks * expect.f_frsize;
    // Other processes running on the operating system may have changed
    // the amount of space available. Check that these are within tolerances.
    // Currently 5% of capacity
    const std::uintmax_t delta = expect_cap / 20;
    const path cases[] = {
        StaticEnv::File,
        StaticEnv::Dir,
        StaticEnv::Dir2,
        StaticEnv::SymlinkToFile,
        StaticEnv::SymlinkToDir
    };
    for (auto& p : cases) {
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        space_info info = space(p, ec);
        TEST_CHECK(!ec);
        TEST_CHECK((expect.f_blocks * expect.f_frsize) == info.capacity);
        TEST_CHECK(EqualDelta((expect.f_bfree  * expect.f_frsize), info.free, delta));
        TEST_CHECK(EqualDelta((expect.f_bavail * expect.f_frsize), info.available, delta));
    }
}

TEST_SUITE_END()
