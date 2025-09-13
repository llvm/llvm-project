//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <fstream>

// streamsize xsputn(const char_type*, streamsize) override;

// This isn't a required override by the standard, but most implementations override it, since it allows for
// significantly improved performance in some cases. All of this code is required to work, so this isn't a libc++
// extension

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <cstring>
#include <fstream>
#include <locale>
#include <vector>

#include "test_macros.h"

typedef std::filebuf::pos_type pos_type;
typedef std::filebuf::off_type off_type;

void sputn_seekoff(char* buf,
                   const size_t buf_size,
                   const std::streamsize chunk_size1,
                   const off_type offset1,
                   const std::streamsize chunk_size2) {
  std::string data{"abcdefghijklmnopqrstuvwxyz"};
  const std::streamsize data_size = static_cast<std::streamsize>(data.size());
  assert(chunk_size1 <= data_size);
  assert(chunk_size2 <= data_size);
  // vector with expected data in the file to be written
  std::size_t result_size = 5 + chunk_size1 + chunk_size2 + 1;
  if (offset1 > 0) {
    result_size += offset1;
  }
  std::vector<char> result(result_size, 0);
  {
    std::filebuf f;
    f.pubsetbuf(buf, buf_size);
    assert(f.open("sputn_seekoff.dat", std::ios_base::out) != 0);
    assert(f.is_open());

    assert(f.pubseekoff(off_type(5), std::ios_base::beg) = off_type(5));

    std::vector<char> chunk(data.begin() + 5, data.begin() + 5 + chunk_size1);
    std::copy(chunk.begin(), chunk.end(), result.begin() + 5);
    const std::streamsize len1 = f.sputn(chunk.data(), chunk_size1);
    assert(len1 == chunk_size1);
    // check that nothing in the original chunk was modified by sputn()
    assert(std::strncmp(chunk.data(), data.substr(5, len1).c_str(), len1) == 0);

    pos_type p1 = f.pubseekoff(offset1, std::ios_base::cur);
    char c;
    if (p1 < 0) {
      p1 = f.pubseekoff(0, std::ios_base::beg);
      assert(p1 == 0);
      c = '^';
    } else {
      assert(p1 == 5 + len1 + offset1);
      if (p1 > data_size) {
        c = '_';
      } else {
        c = data[p1];
      }
    }

    result[p1] = c;
    assert(f.sputc(c) == c);

    f.pubseekpos(std::ios_base::beg);
    result[0] = 'A';
    assert(f.sputc(toupper(data[0])) == 'A');

    pos_type end_pos = f.pubseekoff(off_type(0), std::ios_base::end);
    assert(f.sputc(toupper(data[data_size - 1])) == 'Z');
    result[end_pos] = 'Z';

    assert(f.pubseekpos(p1) == p1);
    result[p1] = toupper(c);
    assert(f.sputc(toupper(c)) == toupper(c));

    pos_type new_pos = result_size - chunk_size2;
    pos_type p2      = f.pubseekoff(new_pos, std::ios_base::beg);
    assert(p2 == new_pos);
    chunk = std::vector<char>(data.end() - chunk_size2, data.end());
    std::copy(chunk.begin(), chunk.end(), result.begin() + p2);
    const std::streamsize len2 = f.sputn(chunk.data(), chunk_size2);
    assert(len2 == chunk_size2);
    assert(std::strncmp(chunk.data(), data.substr(data_size - chunk_size2, chunk_size2).c_str(), len2) == 0);
    f.close();
  }
  std::filebuf f;
  assert(f.open("sputn_seekoff.dat", std::ios_base::in) != 0);
  assert(f.is_open());
  std::vector<char> check(result.size(), -1);
  const std::size_t len = f.sgetn(check.data(), check.size());
  assert(len == result.size());
  for (size_t i = 0; i < len; ++i) {
    assert(check[i] == result[i]);
  }
}

void sputn_not_open() {
  std::vector<char> data(10, 'a');
  std::filebuf f;
  std::streamsize len = f.sputn(data.data(), data.size());
  assert(len == 0);
  assert(std::strncmp(data.data(), "aaaaaaaaaa", 10) == 0);
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
void sputn_not_open_wchar() {
  std::vector<wchar_t> data(10, L'a');
  std::wfilebuf f;
  std::streamsize len = f.sputn(data.data(), data.size());
  assert(len == 0);
  assert(std::wcsncmp(data.data(), L"aaaaaaaaaa", 10) == 0);
}
#endif

int main(int, char**) {
  sputn_not_open();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  sputn_not_open_wchar();
#endif

  sputn_seekoff(nullptr, 10, 21, -27, 1);
  sputn_seekoff(nullptr, 10, 1, -27, 1);
  sputn_seekoff(nullptr, 10, 10, 14, 12);
  sputn_seekoff(nullptr, 10, 1, -2, 1);
  sputn_seekoff(nullptr, 10, 10, -4, 12);
  sputn_seekoff(nullptr, 10, 11, -12, 3);
  sputn_seekoff(nullptr, 10, 7, 3, 8);
  sputn_seekoff(nullptr, 10, 5, -5, 12);
  sputn_seekoff(nullptr, 10, 1, 1, 1);
  sputn_seekoff(nullptr, 10, 9, 0, 1);

  return 0;
}
