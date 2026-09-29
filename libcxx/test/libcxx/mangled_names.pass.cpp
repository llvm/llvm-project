//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We're using `string::starts_with` in this test
// UNSUPPORTED: c++03, c++11, c++14, c++17

// Make sure that the mangling of our public types stays the same

// UNSUPPORTED: no-rtti, msvc

#include <cassert>
#include <charconv>
#include <iostream>
#include <map>
#include <typeinfo>
#include <string>
#include <string_view>

template <class>
struct mangling {};

struct test_struct {};

_LIBCPP_BEGIN_NAMESPACE_STD
struct ns_mangling {};
_LIBCPP_END_NAMESPACE_STD

namespace std::__name {
struct ns_mangling {};
} // namespace std::__name

namespace std::__long_name_to_make_sure_multiple_digits_work {
struct ns_mangling {};
} // namespace std::__long_name_to_make_sure_multiple_digits_work

std::string get_std_inline_namespace_mangling(const std::type_info& info) {
  std::string name = info.name();
  assert(name.starts_with("NSt"));
  unsigned name_len;
  auto res = std::from_chars(name.data() + 3, name.data() + name.size(), name_len);
  assert(res.ec == std::errc{});
  return std::move(name).substr(0, (res.ptr + name_len) - name.data());
}

void expect_mangling(const std::type_info& info, std::string expected_name) {
  if (expected_name != info.name())
    std::__libcpp_verbose_abort("Expected: '%s'\n     Got: '%s'\n", expected_name.c_str(), info.name());
}

// Mangling names are really long, but splitting it up into multiple lines doesn't make it any more readable
// clang-format off
int main(int, char**) {
  // self-test inline namespace recovery
  assert(get_std_inline_namespace_mangling(typeid(std::__name::ns_mangling)) == "NSt6__name");
  assert(get_std_inline_namespace_mangling(typeid(std::__long_name_to_make_sure_multiple_digits_work::ns_mangling)) == "NSt45__long_name_to_make_sure_multiple_digits_work");

  // selftest
  expect_mangling(typeid(test_struct), "11test_struct");

  std::string ns_std = get_std_inline_namespace_mangling(typeid(std::ns_mangling));
  std::string ptrdiff = typeid(std::ptrdiff_t).name();

  // std::map
  expect_mangling(typeid(std::map<int, int>), ns_std + "3mapIiiNS_4lessIiEENS_9allocatorINS_4pairIKiiEEEEEE");
  expect_mangling(typeid(std::map<int, int>::iterator), ns_std + "14__map_iteratorINS_15__tree_iteratorINS_12__value_typeIiiEEPNS_11__tree_nodeIS3_PvEE" + ptrdiff +"EEEE");
  expect_mangling(typeid(std::map<int, int>::const_iterator), ns_std + "20__map_const_iteratorINS_21__tree_const_iteratorINS_12__value_typeIiiEEPNS_11__tree_nodeIS3_PvEE" + ptrdiff + "EEEE");

  return 0;
}
