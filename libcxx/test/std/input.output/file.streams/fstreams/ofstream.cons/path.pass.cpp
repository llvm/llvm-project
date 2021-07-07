//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-filesystem-library

// Filesystem is supported on Apple platforms starting with macosx10.15.
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// <fstream>

// plate <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// explicit basic_ofstream(const filesystem::path& s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <filesystem>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

namespace fs = std::filesystem;

int main(int, char**) {
  fs::path p = get_temp_file_name();
  {
    static_assert(!std::is_convertible<fs::path, std::ofstream>::value,
                  "ctor should be explicit");
    static_assert(std::is_constructible<std::ofstream, fs::path const&,
                                        std::ios_base::openmode>::value,
                  "");
  }
  {
    std::ofstream stream(p);
    stream << 3.25;
  }
  {
    std::ifstream stream(p);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  {
    std::ifstream stream(p, std::ios_base::out);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());
  {
    std::wofstream stream(p);
    stream << 3.25;
  }
  {
    std::wifstream stream(p);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  {
    std::wifstream stream(p, std::ios_base::out);
    double x = 0;
    stream >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());

  return 0;
}
