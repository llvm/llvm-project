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

// class directory_entry

// directory_entry& operator=(directory_entry const&) = default;
// directory_entry& operator=(directory_entry&&) noexcept = default;
// void assign(path const&);
// void replace_filename(path const&);

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>


void test_copy_assign_operator()
{
  using namespace fs;
  // Copy
  {
    static_assert(std::is_copy_assignable<directory_entry>::value,
                  "directory_entry must be copy assignable");
    static_assert(!std::is_nothrow_copy_assignable<directory_entry>::value,
                  "directory_entry's copy assignment cannot be noexcept");
    const path p("foo/bar/baz");
    const path p2("abc");
    const directory_entry e(p);
    directory_entry e2;
    assert(e.path() == p && e2.path() == path());
    e2 = e;
    assert(e.path() == p && e2.path() == p);
    directory_entry e3(p2);
    e2 = e3;
    assert(e2.path() == p2 && e3.path() == p2);
  }
}


void test_move_assign_operator()
{
  using namespace fs;
  // Copy
  {
    static_assert(std::is_nothrow_move_assignable<directory_entry>::value,
                  "directory_entry is noexcept move assignable");
    const path p("foo/bar/baz");
    const path p2("abc");
    directory_entry e(p);
    directory_entry e2(p2);
    assert(e.path() == p && e2.path() == p2);
    e2 = std::move(e);
    assert(e2.path() == p);
    assert(e.path() != p); // testing moved from state
  }
}

void test_path_assign_method()
{
  using namespace fs;
  const path p("foo/bar/baz");
  const path p2("abc");
  directory_entry e(p);
  {
    static_assert(std::is_same<decltype(e.assign(p)), void>::value,
                  "return type should be void");
    static_assert(noexcept(e.assign(p)) == false, "operation must not be noexcept");
  }
  {
    assert(e.path() == p);
    e.assign(p2);
    assert(e.path() == p2 && e.path() != p);
    e.assign(p);
    assert(e.path() == p && e.path() != p2);
  }
}

void test_replace_filename_method()
{
  using namespace fs;
  const path p("/path/to/foo.exe");
  const path replace("bar.out");
  const path expect("/path/to/bar.out");
  directory_entry e(p);
  {
    static_assert(noexcept(e.replace_filename(replace)) == false,
                  "operation cannot be noexcept");
    static_assert(std::is_same<decltype(e.replace_filename(replace)), void>::value,
                  "operation must return void");
  }
  {
    assert(e.path() == p);
    e.replace_filename(replace);
    assert(e.path() == expect);
  }
}

int main() {
  test_copy_assign_operator();
  test_move_assign_operator();
  test_path_assign_method();
  test_replace_filename_method();
}
