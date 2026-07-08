//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIX fopen() does not support the 'x' mode.

// REQUIRES: aix

#include <fstream>

void test() {
  std::filebuf fb;
  fb.open("f", std::ios_base::out | std::ios_base::noreplace); // expected-warning {{fstream::open() with noreplace is not supported on AIX; open() will return failure}}

  std::ifstream ifs;
  ifs.open("f", std::ios_base::in | std::ios_base::noreplace); // expected-warning {{fstream::open() with noreplace is not supported on AIX; open() will return failure}}

  std::ofstream ofs;
  ofs.open("f", std::ios_base::out | std::ios_base::noreplace); // expected-warning {{fstream::open() with noreplace is not supported on AIX; open() will return failure}}

  std::fstream fs;
  fs.open("f", std::ios_base::out | std::ios_base::noreplace); // expected-warning {{fstream::open() with noreplace is not supported on AIX; open() will return failure}}
}
