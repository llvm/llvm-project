//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: c++filesystem-disabled

// <fstream>

// plate <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// void open(const filesystem::path& s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <filesystem>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

namespace fs = std::filesystem;

int main(int, char**) {
  fs::path p = get_temp_file_name();
  {
    std::ofstream fs;
    assert(!fs.is_open());
    char c = 'a';
    fs << c;
    assert(fs.fail());
    fs.open(p);
    assert(fs.is_open());
    fs << c;
  }
  {
    std::ifstream fs(p.c_str());
    char c = 0;
    fs >> c;
    assert(c == 'a');
  }
  std::remove(p.c_str());
  {
    std::wofstream fs;
    assert(!fs.is_open());
    wchar_t c = L'a';
    fs << c;
    assert(fs.fail());
    fs.open(p);
    assert(fs.is_open());
    fs << c;
  }
  {
    std::wifstream fs(p.c_str());
    wchar_t c = 0;
    fs >> c;
    assert(c == L'a');
  }
  std::remove(p.c_str());

  return 0;
}
