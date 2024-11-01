//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// path proximate(const path& p, error_code &ec)
// path proximate(const path& p, const path& base = current_path())
// path proximate(const path& p, const path& base, error_code& ec);

#include <filesystem>
#include <cassert>

#include "assert_macros.h"
#include "concat_macros.h"
#include "test_macros.h"
#include "count_new.h"
#include "filesystem_test_helper.h"
#include "../../class.path/path_helper.h"
namespace fs = std::filesystem;

static int count_path_elems(const fs::path& p) {
  int count = 0;
  for (auto&& elem : p) {
    if (elem != p.root_name() && elem != "/" && elem != "")
      ++count;
  }
  return count;
}

static void signature_test() {
  using fs::path;
  const path p;
  ((void)p);
  std::error_code ec;
  ((void)ec);
  ASSERT_NOT_NOEXCEPT(proximate(p));
  ASSERT_NOT_NOEXCEPT(proximate(p, p));
  ASSERT_NOT_NOEXCEPT(proximate(p, ec));
  ASSERT_NOT_NOEXCEPT(proximate(p, p, ec));
}

static void basic_test() {
  using fs::path;
  const path cwd        = fs::current_path();
  const path parent_cwd = cwd.parent_path();
  const path curdir     = cwd.filename();
  assert(!cwd.native().empty());
  int cwd_depth = count_path_elems(cwd);
  path dot_dot_to_root;
  for (int i = 0; i < cwd_depth; ++i)
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
  for (auto& TC : TestCases) {
    std::error_code ec    = GetTestEC();
    fs::path p            = TC.input;
    const fs::path output = fs::proximate(p, TC.base, ec);
    fs::path expect       = TC.expect;
    expect.make_preferred();
    TEST_REQUIRE(!ec,
                 TEST_WRITE_CONCATENATED(
                     "Input: ", TC.input.string(), "\nBase: ", TC.base.string(), "\nExpected: ", expect.string()));

    const path canon_input = fs::weakly_canonical(TC.input);
    const path canon_base  = fs::weakly_canonical(TC.base);
    const path lexically_p = canon_input.lexically_proximate(canon_base);
    TEST_REQUIRE(
        PathEq(output, expect),
        TEST_WRITE_CONCATENATED(
            "Input: ",
            TC.input.string(),
            "\nBase: ",
            TC.base.string(),
            "\nExpected: ",
            expect.string(),
            "\nOutput: ",
            output.string(),
            "\nLex Prox: ",
            lexically_p.string(),
            "\nCanon Input: ",
            canon_input.string(),
            "\nCanon Base: ",
            canon_base.string()));
  }
}

int main(int, char**) {
  signature_test();
  basic_test();

  return 0;
}
