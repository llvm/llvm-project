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

//          directory_entry() noexcept = default;
//          directory_entry(const directory_entry&) = default;
//          directory_entry(directory_entry&&) noexcept = default;
// explicit directory_entry(const path);

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

namespace fs = std::experimental::filesystem;

void test_default_ctor()
{
  using namespace fs;
  // Default
  {
    static_assert(std::is_nothrow_default_constructible<directory_entry>::value,
                  "directory_entry must have a nothrow default constructor");
    directory_entry e;
    assert(e.path() == path());
  }
}


void test_copy_ctor()
{
  using namespace fs;
  // Copy
  {
    static_assert(std::is_copy_constructible<directory_entry>::value,
                  "directory_entry must be copy constructible");
    static_assert(!std::is_nothrow_copy_constructible<directory_entry>::value,
                  "directory_entry's copy constructor cannot be noexcept");
    const path p("foo/bar/baz");
    const directory_entry e(p);
    assert(e.path() == p);
    directory_entry e2(e);
    assert(e.path() == p);
    assert(e2.path() == p);
  }

}

void test_move_ctor()
{
  using namespace fs;
  // Move
  {
    static_assert(std::is_nothrow_move_constructible<directory_entry>::value,
                  "directory_entry must be nothrow move constructible");
    const path p("foo/bar/baz");
    directory_entry e(p);
    assert(e.path() == p);
    directory_entry e2(std::move(e));
    assert(e2.path() == p);
    assert(e.path()  != p); // Testing moved from state.
  }
}

void test_path_ctor() {
  using namespace fs;
  {
    static_assert(std::is_constructible<directory_entry, const path&>::value,
                  "directory_entry must be constructible from path");
    static_assert(!std::is_nothrow_constructible<directory_entry, const path&>::value,
                  "directory_entry constructor should not be noexcept");
    static_assert(!std::is_convertible<path const&, directory_entry>::value,
                  "directory_entry constructor should be explicit");
  }
  {
    const path p("foo/bar/baz");
    const directory_entry e(p);
    assert(p == e.path());
  }
}

int main() {
  test_default_ctor();
  test_copy_ctor();
  test_move_ctor();
  test_path_ctor();
}
