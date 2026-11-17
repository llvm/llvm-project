//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// QEMU does not detect EOF, when reading from stdin
// "echo -n" suppresses any characters after the output and so the test hangs.
// https://gitlab.com/qemu-project/qemu/-/issues/1963
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

// This test hangs on Android devices that lack shell_v2, which was added in
// Android N (API 24).
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && android-device-api={{2[1-3]}}

// <iostream>

// istream cin;

// std::cin is backed by __stdinbuf, which reads from the C stdin FILE so that
// C++ and C input can be interleaved (std::ios_base::sync_with_stdio). Its
// xsgetn() takes a bulk fread() fast path for the no-conversion case, while
// single-character operations go through __getchar()/getc(). This test checks
// that the two paths observe the same stream and agree on the putback
// bookkeeping (__last_consumed_). Some of the putback expectations
// (e.g. sungetc() failing after re-consuming a putback character) are
// libc++-specific, which is why this test lives under test/libcxx.

// RUN: %{build}
// RUN: echo -n ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 > %t.input
// RUN: %{exec} %t.exe < %t.input

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>

int main(int, char**) {
  typedef std::char_traits<char> Traits;
  char buf[16];

  // C read first: bulk reads must continue where stdio left off.
  int c = std::getchar();
  assert(c == 'A');

  // A character pushed back with ungetc() onto the FILE must be the first
  // byte a bulk read delivers.
  assert(std::ungetc('a', stdin) == 'a');
  std::cin.read(buf, 4);
  assert(std::cin.gcount() == 4);
  assert(std::memcmp(buf, "aBCD", 4) == 0);

  // peek() (underflow, which pushes the byte back with ungetc()) followed by
  // a bulk read must not lose or duplicate the peeked byte.
  assert(std::cin.peek() == 'E');
  std::cin.read(buf, 3);
  assert(std::cin.gcount() == 3);
  assert(std::memcmp(buf, "EFG", 3) == 0);

  // sungetc() after a bulk read: the last character delivered by the bulk
  // read must be the one made available again.
  assert(std::cin.rdbuf()->sungetc() == 'G');
  std::cin.read(buf, 2);
  assert(std::cin.gcount() == 2);
  assert(std::memcmp(buf, "GH", 2) == 0);

  // putback() of an arbitrary character, then a bulk read: the pending
  // character comes first, the rest comes from the stream.
  assert(std::cin.putback('h').good());
  std::cin.read(buf, 2);
  assert(std::cin.gcount() == 2);
  assert(std::memcmp(buf, "hI", 2) == 0);

  // A bulk read that delivers ONLY a pending putback character must behave
  // like __getchar() re-consuming it: sungetc() afterwards fails. This
  // matches the single-character path, which forgets __last_consumed_ when
  // returning a putback character.
  assert(std::cin.putback('q').good());
  std::cin.read(buf, 1);
  assert(std::cin.gcount() == 1);
  assert(buf[0] == 'q');
  assert(std::cin.rdbuf()->sungetc() == Traits::eof());

  // Back to C stdio: it must see the byte right after what C++ consumed.
  c = std::getchar();
  assert(c == 'J');

  // Single-character get() still works after bulk reads.
  assert(std::cin.get() == 'K');

  // Read to EOF: 'L'..'Z' and '0'..'9' remain (25 characters). The first
  // read is satisfied in full, the second is short and hits EOF.
  std::cin.read(buf, sizeof(buf));
  assert(std::cin.gcount() == 16);
  std::cin.read(buf, sizeof(buf));
  assert(std::cin.gcount() == 9);
  assert(std::cin.eof());

  return 0;
}
