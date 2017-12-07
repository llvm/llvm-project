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


#include <experimental/filesystem>
#include <type_traits>
#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;
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
    , {"/", {"/"}, "/", "", "/", "", "", "/"}
    , {"/foo", {"/", "foo"}, "/", "", "/", "foo", "/", "foo"}
    , {"foo/", {"foo", "."}, "", "", "", "foo/", "foo", "."}
    , {"/foo/", {"/", "foo", "."}, "/", "", "/", "foo/", "/foo", "."}
    , {"foo/bar", {"foo","bar"}, "",  "", "",  "foo/bar", "foo", "bar"}
    , {"/foo//bar", {"/","foo","bar"}, "/", "", "/", "foo/bar", "/foo", "bar"}
    , {"//net", {"//net"}, "//net", "//net", "", "", "", "//net"}
    , {"//net/foo", {"//net", "/", "foo"}, "//net/", "//net", "/", "foo", "//net/", "foo"}
    , {"///foo///", {"/", "foo", "."}, "/", "", "/", "foo///", "///foo", "."}
    , {"///foo///bar", {"/", "foo", "bar"}, "/", "", "/", "foo///bar", "///foo", "bar"}
    , {"/.", {"/", "."}, "/", "", "/", ".", "/", "."}
    , {"./", {".", "."}, "", "", "", "./", ".", "."}
    , {"/..", {"/", ".."}, "/", "", "/", "..", "/", ".."}
    , {"../", {"..", "."}, "", "", "", "../", "..", "."}
    , {"foo/.", {"foo", "."}, "", "", "", "foo/.", "foo", "."}
    , {"foo/..", {"foo", ".."}, "", "", "", "foo/..", "foo", ".."}
    , {"foo/./", {"foo", ".", "."}, "", "", "", "foo/./", "foo/.", "."}
    , {"foo/./bar", {"foo", ".", "bar"}, "", "", "", "foo/./bar", "foo/.", "bar"}
    , {"foo/../", {"foo", "..", "."}, "", "", "", "foo/../", "foo/..", "."}
    , {"foo/../bar", {"foo", "..", "bar"}, "", "", "", "foo/../bar", "foo/..", "bar"}
    , {"c:", {"c:"}, "", "", "", "c:", "", "c:"}
    , {"c:/", {"c:", "."}, "", "", "", "c:/", "c:", "."}
    , {"c:foo", {"c:foo"}, "", "", "", "c:foo", "", "c:foo"}
    , {"c:/foo", {"c:", "foo"}, "", "", "", "c:/foo", "c:", "foo"}
    , {"c:foo/", {"c:foo", "."}, "", "", "", "c:foo/", "c:foo", "."}
    , {"c:/foo/", {"c:", "foo", "."}, "", "", "", "c:/foo/",  "c:/foo", "."}
    , {"c:/foo/bar", {"c:", "foo", "bar"}, "", "", "", "c:/foo/bar", "c:/foo", "bar"}
    , {"prn:", {"prn:"}, "", "", "", "prn:", "", "prn:"}
    , {"c:\\", {"c:\\"}, "", "", "", "c:\\", "", "c:\\"}
    , {"c:\\foo", {"c:\\foo"}, "", "", "", "c:\\foo", "", "c:\\foo"}
    , {"c:foo\\", {"c:foo\\"}, "", "", "", "c:foo\\", "", "c:foo\\"}
    , {"c:\\foo\\", {"c:\\foo\\"}, "", "", "", "c:\\foo\\", "", "c:\\foo\\"}
    , {"c:\\foo/",  {"c:\\foo", "."}, "", "", "", "c:\\foo/", "c:\\foo", "."}
    , {"c:/foo\\bar", {"c:", "foo\\bar"}, "", "", "", "c:/foo\\bar", "c:", "foo\\bar"}
    , {"//", {"//"}, "//", "//", "", "", "", "//"}
  };

void decompPathTest()
{
  using namespace fs;
  for (auto const & TC : PathTestCases) {
    path p(TC.raw);
    assert(p == TC.raw);

    assert(p.root_path() == TC.root_path);
    assert(p.has_root_path() !=  TC.root_path.empty());

    assert(p.root_name() == TC.root_name);
    assert(p.has_root_name() !=  TC.root_name.empty());

    assert(p.root_directory() == TC.root_directory);
    assert(p.has_root_directory() !=  TC.root_directory.empty());

    assert(p.relative_path() == TC.relative_path);
    assert(p.has_relative_path() !=  TC.relative_path.empty());

    assert(p.parent_path() == TC.parent_path);
    assert(p.has_parent_path() !=  TC.parent_path.empty());

    assert(p.filename() == TC.filename);
    assert(p.has_filename() !=  TC.filename.empty());

    assert(p.is_absolute() == p.has_root_directory());
    assert(p.is_relative() !=  p.is_absolute());

    assert(checkCollectionsEqual(p.begin(), p.end(),
                                 TC.elements.begin(), TC.elements.end()));
    // check backwards

    std::vector<fs::path> Parts;
    for (auto it = p.end(); it != p.begin(); )
      Parts.push_back(*--it);
    assert(checkCollectionsEqual(Parts.begin(), Parts.end(),
                                 TC.elements.rbegin(), TC.elements.rend()));
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
  , {"/", "/", "/", ""}
  , {"foo", "foo", "foo", ""}
  , {"/foo/bar.txt", "bar.txt", "bar", ".txt"}
  , {"foo..txt", "foo..txt", "foo.", ".txt"}
};


void decompFilenameTest()
{
  using namespace fs;
  for (auto const & TC : FilenameTestCases) {
    path p(TC.raw);
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

int main()
{
  decompPathTest();
  decompFilenameTest();
}
