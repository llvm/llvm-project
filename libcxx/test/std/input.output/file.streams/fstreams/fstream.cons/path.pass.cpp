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
// UNSUPPORTED: with_system_cxx_lib=macosx10.14
// UNSUPPORTED: with_system_cxx_lib=macosx10.13
// UNSUPPORTED: with_system_cxx_lib=macosx10.12
// UNSUPPORTED: with_system_cxx_lib=macosx10.11
// UNSUPPORTED: with_system_cxx_lib=macosx10.10
// UNSUPPORTED: with_system_cxx_lib=macosx10.9

// <fstream>

// plate <class charT, class traits = char_traits<charT> >
// class basic_fstream

// explicit basic_fstream(const filesystem::path& s,
//     ios_base::openmode mode = ios_base::in|ios_base::out);

#include <fstream>
#include <filesystem>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

namespace fs = std::filesystem;

int main(int, char**) {
  fs::path p = get_temp_file_name();
  {
    std::fstream fs(p, std::ios_base::in | std::ios_base::out |
                           std::ios_base::trunc);
    double x = 0;
    fs << 3.25;
    fs.seekg(0);
    fs >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());
  {
    std::wfstream fs(p, std::ios_base::in | std::ios_base::out |
                            std::ios_base::trunc);
    double x = 0;
    fs << 3.25;
    fs.seekg(0);
    fs >> x;
    assert(x == 3.25);
  }
  std::remove(p.string().c_str());

  return 0;
}
