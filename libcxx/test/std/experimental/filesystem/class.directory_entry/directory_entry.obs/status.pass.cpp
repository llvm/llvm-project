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

// file_status status() const;
// file_status status(error_code const&) const noexcept;

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.hpp"

int main()
{
  using namespace fs;
  {
    const directory_entry e("foo");
    std::error_code ec;
    static_assert(std::is_same<decltype(e.status()), file_status>::value, "");
    static_assert(std::is_same<decltype(e.status(ec)), file_status>::value, "");
    static_assert(noexcept(e.status()) == false, "");
    static_assert(noexcept(e.status(ec)) == true, "");
  }
  auto TestFn = [](path const& p) {
    const directory_entry e(p);
    std::error_code pec, eec;
    file_status ps = fs::status(p, pec);
    file_status es = e.status(eec);
    assert(ps.type() == es.type());
    assert(ps.permissions() == es.permissions());
    assert(pec == eec);
  };
  {
    TestFn(StaticEnv::File);
    TestFn(StaticEnv::Dir);
    TestFn(StaticEnv::SymlinkToFile);
    TestFn(StaticEnv::DNE);
  }
}
