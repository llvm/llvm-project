//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// 8.4.9 path decomposition [path.decompose]
//------------------------------------------
// path root_name() const;
// path root_directory() const;
// path root_path() const;
// path relative_path() const;
// path parent_path() const;
// path filename() const;
// path stem() const;
// path extension() const;
//-------------------------------
// 8.4.10 path query [path.query]
//-------------------------------
// bool empty() const noexcept;
// bool has_root_path() const;
// bool has_root_name() const;
// bool has_root_directory() const;
// bool has_relative_path() const;
// bool has_parent_path() const;
// bool has_filename() const;
// bool has_stem() const;
// bool has_extension() const;
// bool is_absolute() const;
// bool is_relative() const;
//-------------------------------
// 8.5 path iterators [path.itr]
//-------------------------------
// iterator begin() const;
// iterator end() const;


#include "filesystem_include.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"

struct ComparePathExact {
  bool operator()(fs::path const& LHS, std::string const& RHS) const {
    return LHS.string() == RHS;
  }
};

struct PathDecomposeTestcase
{
    std::string raw;
    std::vector<std::string> elements;
    std::string root_path;
    std::string root_name;
    std::string root_directory;
    std::string relative_path;
    std::string parent_path;
    std::string filename;
};

const PathDecomposeTestcase PathTestCases[] =
  {
      {"", {}, "", "", "", "", "", ""}
    , {".", {"."}, "", "", "", ".", "", "."}
    , {"..", {".."}, "", "", "", "..", "", ".."}
    , {"foo", {"foo"}, "", "", "", "foo", "", "foo"}
    , {"/", {"/"}, "/", "", "/", "", "/", ""}
    , {"/foo", {"/", "foo"}, "/", "", "/", "foo", "/", "foo"}
    , {"foo/", {"foo", ""}, "", "", "", "foo/", "foo", ""}
    , {"/foo/", {"/", "foo", ""}, "/", "", "/", "foo/", "/foo", ""}
    , {"foo/bar", {"foo","bar"}, "",  "", "",  "foo/bar", "foo", "bar"}
    , {"/foo//bar", {"/","foo","bar"}, "/", "", "/", "foo/bar", "/foo", "bar"}
#ifdef _WIN32
    , {"//net", {"//net"}, "//net", "//net", "", "", "//net", ""}
    , {"//net/", {"//net", "/"}, "//net/", "//net", "/", "", "//net/", ""}
    , {"//net/foo", {"//net", "/", "foo"}, "//net/", "//net", "/", "foo", "//net/", "foo"}
#else
    , {"//net", {"/", "net"}, "/", "", "/", "net", "/", "net"}
    , {"//net/", {"/", "net", ""}, "/", "", "/", "net/", "//net", ""}
    , {"//net/foo", {"/", "net", "foo"}, "/", "", "/", "net/foo", "/net", "foo"}
#endif
    , {"///foo///", {"/", "foo", ""}, "/", "", "/", "foo///", "///foo", ""}
    , {"///foo///bar", {"/", "foo", "bar"}, "/", "", "/", "foo///bar", "///foo", "bar"}
    , {"/.", {"/", "."}, "/", "", "/", ".", "/", "."}
    , {"./", {".", ""}, "", "", "", "./", ".", ""}
    , {"/..", {"/", ".."}, "/", "", "/", "..", "/", ".."}
    , {"../", {"..", ""}, "", "", "", "../", "..", ""}
    , {"foo/.", {"foo", "."}, "", "", "", "foo/.", "foo", "."}
    , {"foo/..", {"foo", ".."}, "", "", "", "foo/..", "foo", ".."}
    , {"foo/./", {"foo", ".", ""}, "", "", "", "foo/./", "foo/.", ""}
    , {"foo/./bar", {"foo", ".", "bar"}, "", "", "", "foo/./bar", "foo/.", "bar"}
    , {"foo/../", {"foo", "..", ""}, "", "", "", "foo/../", "foo/..", ""}
    , {"foo/../bar", {"foo", "..", "bar"}, "", "", "", "foo/../bar", "foo/..", "bar"}
#ifdef _WIN32
    , {"c:", {"c:"}, "c:", "c:", "", "", "c:", ""}
    , {"c:/", {"c:", "/"}, "c:/", "c:", "/", "", "c:/", ""}
    , {"c:foo", {"c:", "foo"}, "c:", "c:", "", "foo", "c:", "foo"}
    , {"c:/foo", {"c:", "/", "foo"}, "c:/", "c:", "/", "foo", "c:/", "foo"}
    , {"c:foo/", {"c:", "foo", ""}, "c:", "c:", "", "foo/", "c:foo", ""}
    , {"c:/foo/", {"c:", "/", "foo", ""}, "c:/", "c:", "/", "foo/",  "c:/foo", ""}
    , {"c:/foo/bar", {"c:", "/", "foo", "bar"}, "c:/", "c:", "/", "foo/bar", "c:/foo", "bar"}
#else
    , {"c:", {"c:"}, "", "", "", "c:", "", "c:"}
    , {"c:/", {"c:", ""}, "", "", "", "c:/", "c:", ""}
    , {"c:foo", {"c:foo"}, "", "", "", "c:foo", "", "c:foo"}
    , {"c:/foo", {"c:", "foo"}, "", "", "", "c:/foo", "c:", "foo"}
    , {"c:foo/", {"c:foo", ""}, "", "", "", "c:foo/", "c:foo", ""}
    , {"c:/foo/", {"c:", "foo", ""}, "", "", "", "c:/foo/",  "c:/foo", ""}
    , {"c:/foo/bar", {"c:", "foo", "bar"}, "", "", "", "c:/foo/bar", "c:/foo", "bar"}
#endif
    , {"prn:", {"prn:"}, "", "", "", "prn:", "", "prn:"}
#ifdef _WIN32
    , {"c:\\", {"c:", "\\"}, "c:\\", "c:", "\\", "", "c:\\", ""}
    , {"c:\\foo", {"c:", "\\", "foo"}, "c:\\", "c:", "\\", "foo", "c:\\", "foo"}
    , {"c:foo\\", {"c:", "foo", ""}, "c:", "c:", "", "foo\\", "c:foo", ""}
    , {"c:\\foo\\", {"c:", "\\", "foo", ""}, "c:\\", "c:", "\\", "foo\\", "c:\\foo", ""}
    , {"c:\\foo/",  {"c:", "\\", "foo", ""}, "c:\\", "c:", "\\", "foo/", "c:\\foo", ""}
    , {"c:/foo\\bar", {"c:", "/", "foo", "bar"}, "c:\\", "c:", "\\", "foo\\bar", "c:/foo", "bar"}
#else
    , {"c:\\", {"c:\\"}, "", "", "", "c:\\", "", "c:\\"}
    , {"c:\\foo", {"c:\\foo"}, "", "", "", "c:\\foo", "", "c:\\foo"}
    , {"c:foo\\", {"c:foo\\"}, "", "", "", "c:foo\\", "", "c:foo\\"}
    , {"c:\\foo\\", {"c:\\foo\\"}, "", "", "", "c:\\foo\\", "", "c:\\foo\\"}
    , {"c:\\foo/",  {"c:\\foo", ""}, "", "", "", "c:\\foo/", "c:\\foo", ""}
    , {"c:/foo\\bar", {"c:", "foo\\bar"}, "", "", "", "c:/foo\\bar", "c:", "foo\\bar"}
#endif
    , {"//", {"/"}, "/", "", "/", "", "/", ""}
  };

void decompPathTest()
{
  using namespace fs;
  for (auto const & TC : PathTestCases) {
    fs::path p(TC.raw);
    assert(p == TC.raw);

    assert(p.root_path() == TC.root_path);
    assert(p.has_root_path() != TC.root_path.empty());

#ifndef _WIN32
    assert(p.root_name().native().empty());
#endif
    assert(p.root_name() == TC.root_name);
    assert(p.has_root_name() != TC.root_name.empty());

    assert(p.root_directory() == TC.root_directory);
    assert(p.has_root_directory() != TC.root_directory.empty());

    assert(p.relative_path() == TC.relative_path);
    assert(p.has_relative_path() != TC.relative_path.empty());

    assert(p.parent_path() == TC.parent_path);
    assert(p.has_parent_path() != TC.parent_path.empty());

    assert(p.filename() == TC.filename);
    assert(p.has_filename() != TC.filename.empty());

#ifdef _WIN32
    if (!p.has_root_name()) {
      assert(p.is_absolute() == false);
    } else {
      std::string root_name = p.root_name().string();
      assert(root_name.length() >= 2);
      if (root_name[1] == ':') {
        // Drive letter, absolute if has a root directory
        assert(p.is_absolute() == p.has_root_directory());
      } else {
        // Possibly a server path
        // Convert to one separator style, for simplicity
        std::replace(root_name.begin(), root_name.end(), '\\', '/');
        if (root_name[0] == '/' && root_name[1] == '/')
          assert(p.is_absolute() == true);
        else
          assert(p.is_absolute() == false);
      }
    }
#else
    assert(p.is_absolute() == p.has_root_directory());
#endif
    assert(p.is_relative() != p.is_absolute());
    if (p.empty())
      assert(p.is_relative());

    assert(static_cast<std::size_t>(std::distance(p.begin(), p.end())) == TC.elements.size());
    assert(std::equal(p.begin(), p.end(), TC.elements.begin(), ComparePathExact()));

    // check backwards
    std::vector<fs::path> Parts;
    for (auto it = p.end(); it != p.begin(); )
      Parts.push_back(*--it);
    assert(static_cast<std::size_t>(std::distance(Parts.begin(), Parts.end())) == TC.elements.size());
    assert(std::equal(Parts.begin(), Parts.end(), TC.elements.rbegin(), ComparePathExact()));
  }
}


struct FilenameDecompTestcase
{
  std::string raw;
  std::string filename;
  std::string stem;
  std::string extension;
};

const FilenameDecompTestcase FilenameTestCases[] =
{
    {"", "", "", ""}
  , {".", ".", ".", ""}
  , {"..", "..", "..", ""}
  , {"/", "", "", ""}
  , {"foo", "foo", "foo", ""}
  , {"/foo/bar.txt", "bar.txt", "bar", ".txt"}
  , {"foo..txt", "foo..txt", "foo.", ".txt"}
  , {".profile", ".profile", ".profile", ""}
  , {".profile.txt", ".profile.txt", ".profile", ".txt"}
};


void decompFilenameTest()
{
  using namespace fs;
  for (auto const & TC : FilenameTestCases) {
    fs::path p(TC.raw);
    assert(p == TC.raw);
    ASSERT_NOEXCEPT(p.empty());

    assert(p.filename() == TC.filename);
    assert(p.has_filename() != TC.filename.empty());

    assert(p.stem() == TC.stem);
    assert(p.has_stem() != TC.stem.empty());

    assert(p.extension() == TC.extension);
    assert(p.has_extension() != TC.extension.empty());
  }
}

int main(int, char**)
{
  decompPathTest();
  decompFilenameTest();

  return 0;
}
