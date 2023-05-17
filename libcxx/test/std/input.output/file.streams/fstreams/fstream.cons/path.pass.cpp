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

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
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
#endif

  return 0;
}
