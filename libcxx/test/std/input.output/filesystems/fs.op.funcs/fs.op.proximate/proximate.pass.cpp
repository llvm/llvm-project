//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-DEBUG-FIXME

// UNSUPPORTED: c++03

// <filesystem>

// path proximate(const path& p, error_code &ec)
// path proximate(const path& p, const path& base = current_path())
// path proximate(const path& p, const path& base, error_code& ec);

#include "filesystem_include.h"
#include <cassert>
#include <cstdio>

#include "test_macros.h"
#include "count_new.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"


static int count_path_elems(const fs::path& p) {
  int count = 0;
  for (auto&& elem : p) {
    if (elem != p.root_name() && elem != "/" && elem != "")
      ++count;
  }
  return count;
}

TEST_SUITE(filesystem_proximate_path_test_suite)


TEST_CASE(signature_test)
{
    using fs::path;
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(proximate(p));
    ASSERT_NOT_NOEXCEPT(proximate(p, p));
    ASSERT_NOT_NOEXCEPT(proximate(p, ec));
    ASSERT_NOT_NOEXCEPT(proximate(p, p, ec));
}

TEST_CASE(basic_test) {
  using fs::path;
  const path cwd = fs::current_path();
  const path parent_cwd = cwd.parent_path();
  const path curdir = cwd.filename();
  TEST_REQUIRE(!cwd.native().empty());
  int cwd_depth = count_path_elems(cwd);
  path dot_dot_to_root;
  for (int i=0; i < cwd_depth; ++i)
    dot_dot_to_root /= "..";
  path relative_cwd = cwd.native().substr(cwd.root_path().native().size());
  // clang-format off
  struct {
    fs::path input;
    fs::path base;
    fs::path expect;
  } TestCases[] = {
      {"", "", "."},
      {cwd, "a", ".."},
      {parent_cwd, "a", "../.."},
      {"a", cwd, "a"},
      {"a", parent_cwd, curdir / "a"},
      {"/", "a", dot_dot_to_root / ".."},
      {"/", "a/b", dot_dot_to_root / "../.."},
      {"/", "a/b/", dot_dot_to_root / "../.."},
      {"a", "/", relative_cwd / "a"},
      {"a/b", "/", relative_cwd / "a/b"},
      {"a", "/net", ".." / relative_cwd / "a"},
#ifdef _WIN32
      {"//foo/", "//foo", "//foo/"},
      {"//foo", "//foo/", "//foo"},
#else
      {"//foo/", "//foo", "."},
      {"//foo", "//foo/", "."},
#endif
      {"//foo", "//foo", "."},
      {"//foo/", "//foo/", "."},
#ifdef _WIN32
      {"//foo", "a", "//foo"},
      {"//foo/a", "//bar", "//foo/a"},
      {"//foo/a", "//bar/", "//foo/a"},
      {"//foo/a", "b", "//foo/a"},
      {"//foo/a", "/b", "//foo/a"},
      {"//foo/a", "//bar/b", "//foo/a"},
      // Using X: instead of C: to avoid influence from the CWD being under C:
      {"X:/a", "X:/b", "../a"},
      {"X:/a", "X:b", "X:/a"},
      {"X:/a", "Y:/a", "X:/a"},
      {"X:/a", "Y:/b", "X:/a"},
      {"X:/a", "Y:b", "X:/a"},
      {"X:a", "X:/b", "X:a"},
      {"X:a", "X:b", "../a"},
      {"X:a", "Y:/a", "X:a"},
      {"X:a", "Y:/b", "X:a"},
      {"X:a", "Y:b", "X:a"},
#else
      {"//foo", "a", dot_dot_to_root / "../foo"},
      {"//foo/a", "//bar", "../foo/a"},
      {"//foo/a", "//bar/", "../foo/a"},
      {"//foo/a", "b", dot_dot_to_root / "../foo/a"},
      {"//foo/a", "/b", "../foo/a"},
      {"//foo/a", "//bar/b", "../../foo/a"},
      {"X:/a", "X:/b", "../a"},
      {"X:/a", "X:b", "../X:/a"},
      {"X:/a", "Y:/a", "../../X:/a"},
      {"X:/a", "Y:/b", "../../X:/a"},
      {"X:/a", "Y:b", "../X:/a"},
      {"X:a", "X:/b", "../../X:a"},
      {"X:a", "X:b", "../X:a"},
      {"X:a", "Y:/a", "../../X:a"},
      {"X:a", "Y:/b", "../../X:a"},
      {"X:a", "Y:b", "../X:a"},
#endif
      {"a", "a", "."},
      {"a/b", "a/b", "."},
      {"a/b/c/", "a/b/c/", "."},
      {"//foo/a/b", "//foo/a/b", "."},
      {"/a/d", "/a/b/c", "../../d"},
      {"/a/b/c", "/a/d", "../b/c"},
      {"a/b/c", "a", "b/c"},
      {"a/b/c", "a/b/c/x/y", "../.."},
      {"a/b/c", "a/b/c", "."},
      {"a/b", "c/d", "../../a/b"}
  };
  // clang-format on
  int ID = 0;
  for (auto& TC : TestCases) {
    ++ID;
    std::error_code ec = GetTestEC();
    fs::path p = TC.input;
    const fs::path output = fs::proximate(p, TC.base, ec);
    fs::path expect = TC.expect;
    expect.make_preferred();
    if (ec) {
      TEST_CHECK(!ec);
      std::fprintf(stderr, "TEST CASE #%d FAILED:\n"
                  "  Input: '%s'\n"
                  "  Base: '%s'\n"
                  "  Expected: '%s'\n",
        ID, TC.input.string().c_str(), TC.base.string().c_str(),
        expect.string().c_str());
    } else if (!PathEq(output, expect)) {
      TEST_CHECK(PathEq(output, expect));

      const path canon_input = fs::weakly_canonical(TC.input);
      const path canon_base = fs::weakly_canonical(TC.base);
      const path lexically_p = canon_input.lexically_proximate(canon_base);
      std::fprintf(stderr, "TEST CASE #%d FAILED:\n"
                  "  Input: '%s'\n"
                  "  Base: '%s'\n"
                  "  Expected: '%s'\n"
                  "  Output: '%s'\n"
                  "  Lex Prox: '%s'\n"
                  "  Canon Input: '%s'\n"
                  "  Canon Base: '%s'\n",
        ID, TC.input.string().c_str(), TC.base.string().c_str(),
        expect.string().c_str(), output.string().c_str(),
        lexically_p.string().c_str(), canon_input.string().c_str(),
        canon_base.string().c_str());
    }
  }
}

TEST_SUITE_END()
