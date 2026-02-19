//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-variable

#include <filesystem>

// clang-format off

namespace fs = std::filesystem;

fs::path& test() {
  fs::path p;
  char arr[] = "Banane";

  auto&& v1 = fs::path().native(); // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
  auto v2   = fs::path().c_str();  // expected-warning {{temporary whose address is used as value of local variable 'v2' will be destroyed at the end of the full-expression}}

  return p.assign("");                             // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.assign(std::string());                  // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.assign(std::begin(arr), std::end(arr)); // expected-warning {{reference to stack memory associated with local variable 'p' returned}}

  return p.append("");                             // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.append(std::begin(arr), std::end(arr)); // expected-warning {{reference to stack memory associated with local variable 'p' returned}}

  return p.concat("");                             // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.concat(std::begin(arr), std::end(arr)); // expected-warning {{reference to stack memory associated with local variable 'p' returned}}

  return p.make_preferred();     // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.remove_filename();    // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.replace_filename(""); // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
  return p.replace_extension();  // expected-warning {{reference to stack memory associated with local variable 'p' returned}}
}
