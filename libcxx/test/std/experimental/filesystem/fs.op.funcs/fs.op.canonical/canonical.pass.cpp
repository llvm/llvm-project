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

// path canonical(const path& p, const path& base = current_path());
// path canonical(const path& p, error_code& ec);
// path canonical(const path& p, const path& base, error_code& ec);

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

TEST_SUITE(filesystem_canonical_path_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(canonical(p));
    ASSERT_NOT_NOEXCEPT(canonical(p, p));
    ASSERT_NOT_NOEXCEPT(canonical(p, ec));
    ASSERT_NOT_NOEXCEPT(canonical(p, p, ec));
}

// There are 4 cases is the proposal for absolute path.
// Each scope tests one of the cases.
TEST_CASE(test_canonical)
{
    // has_root_name() && has_root_directory()
    const path Root = StaticEnv::Root;
    const path RootName = Root.filename();
    const path DirName = StaticEnv::Dir.filename();
    const path SymlinkName = StaticEnv::SymlinkToFile.filename();
    struct TestCase {
        path p;
        path expect;
        path base = StaticEnv::Root;
    };
    const TestCase testCases[] = {
        { ".", Root, Root},
        { DirName / ".." / "." / DirName, StaticEnv::Dir, Root},
        { StaticEnv::Dir2 / "..",    StaticEnv::Dir },
        { StaticEnv::Dir3 / "../..", StaticEnv::Dir },
        { StaticEnv::Dir / ".",      StaticEnv::Dir },
        { Root / "." / DirName / ".." / DirName, StaticEnv::Dir},
        { path("..") / "." / RootName / DirName / ".." / DirName, StaticEnv::Dir, Root},
        { StaticEnv::SymlinkToFile,  StaticEnv::File },
        { SymlinkName, StaticEnv::File, StaticEnv::Root}
    };
    for (auto& TC : testCases) {
        std::error_code ec;
        const path ret = canonical(TC.p, TC.base, ec);
        TEST_REQUIRE(!ec);
        const path ret2 = canonical(TC.p, TC.base);
        TEST_CHECK(ret == TC.expect);
        TEST_CHECK(ret == ret2);
        TEST_CHECK(ret.is_absolute());
    }
}

TEST_CASE(test_dne_path)
{
    std::error_code ec;
    {
        const path ret = canonical(StaticEnv::DNE, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(ret == path{});
    }
    ec.clear();
    {
        const path ret = canonical(StaticEnv::DNE, StaticEnv::Root, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(ret == path{});
    }
    {
        TEST_CHECK_THROW(filesystem_error, canonical(StaticEnv::DNE));
        TEST_CHECK_THROW(filesystem_error, canonical(StaticEnv::DNE, StaticEnv::Root));
    }
}

TEST_CASE(test_exception_contains_paths)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    const path p = "blabla/dne";
    const path base = StaticEnv::Root;
    try {
        canonical(p, base);
        TEST_REQUIRE(false);
    } catch (filesystem_error const& err) {
        TEST_CHECK(err.path1() == p);
        TEST_CHECK(err.path2() == base);
    }
    try {
        canonical(p);
        TEST_REQUIRE(false);
    } catch (filesystem_error const& err) {
        TEST_CHECK(err.path1() == p);
        TEST_CHECK(err.path2() == current_path());
    }
#endif
}

TEST_SUITE_END()
