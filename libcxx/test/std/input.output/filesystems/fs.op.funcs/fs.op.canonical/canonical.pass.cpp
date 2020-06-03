//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// path canonical(const path& p);
// path canonical(const path& p, error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_canonical_path_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(canonical(p));
    ASSERT_NOT_NOEXCEPT(canonical(p, ec));
}

// There are 4 cases is the proposal for absolute path.
// Each scope tests one of the cases.
TEST_CASE(test_canonical)
{
    static_test_env static_env;
    CWDGuard guard;
    // has_root_name() && has_root_directory()
    const path Root = static_env.Root;
    const path RootName = Root.filename();
    const path DirName = static_env.Dir.filename();
    const path SymlinkName = static_env.SymlinkToFile.filename();
    struct TestCase {
        path p;
        path expect;
        path base;
        TestCase(path p1, path e, path b)
            : p(p1), expect(e), base(b) {}
    };
    const TestCase testCases[] = {
        { ".", Root, Root },
        { DirName / ".." / "." / DirName, static_env.Dir, Root },
        { static_env.Dir2 / "..",    static_env.Dir, Root },
        { static_env.Dir3 / "../..", static_env.Dir, Root },
        { static_env.Dir / ".",      static_env.Dir, Root },
        { Root / "." / DirName / ".." / DirName, static_env.Dir, Root },
        { path("..") / "." / RootName / DirName / ".." / DirName,
          static_env.Dir,
          Root },
        { static_env.SymlinkToFile,  static_env.File, Root },
        { SymlinkName, static_env.File, Root}
    };
    for (auto& TC : testCases) {
        std::error_code ec = GetTestEC();
        fs::current_path(TC.base);
        const path ret = canonical(TC.p, ec);
        TEST_REQUIRE(!ec);
        const path ret2 = canonical(TC.p);
        TEST_CHECK(PathEq(ret, TC.expect));
        TEST_CHECK(PathEq(ret, ret2));
        TEST_CHECK(ret.is_absolute());
    }
}

TEST_CASE(test_dne_path)
{
    static_test_env static_env;
    std::error_code ec = GetTestEC();
    {
        const path ret = canonical(static_env.DNE, ec);
        TEST_CHECK(ec != GetTestEC());
        TEST_REQUIRE(ec);
        TEST_CHECK(ret == path{});
    }
    {
        TEST_CHECK_THROW(filesystem_error, canonical(static_env.DNE));
    }
}

TEST_CASE(test_exception_contains_paths)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    static_test_env static_env;
    CWDGuard guard;
    const path p = "blabla/dne";
    try {
        canonical(p);
        TEST_REQUIRE(false);
    } catch (filesystem_error const& err) {
        TEST_CHECK(err.path1() == p);
        // libc++ provides the current path as the second path in the exception
        LIBCPP_ONLY(TEST_CHECK(err.path2() == current_path()));
    }
    fs::current_path(static_env.Dir);
    try {
        canonical(p);
        TEST_REQUIRE(false);
    } catch (filesystem_error const& err) {
        TEST_CHECK(err.path1() == p);
        LIBCPP_ONLY(TEST_CHECK(err.path2() == static_env.Dir));
    }
#endif
}

TEST_SUITE_END()
