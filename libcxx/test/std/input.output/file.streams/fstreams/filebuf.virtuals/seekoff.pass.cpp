//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: seekoff.dat

// <fstream>

// pos_type seekoff(off_type off, ios_base::seekdir way,
//                  ios_base::openmode which = ios_base::in | ios_base::out);
// pos_type seekpos(pos_type sp,
//                  ios_base::openmode which = ios_base::in | ios_base::out);

#include <cassert>
#include <fstream>

#include "make_string.h"
#include "test_macros.h"

template <class CharT>
void reopen(std::basic_filebuf<CharT>& fb) {
  fb.close();
  fb.open("seekoff.dat", std::ios::in);
}

template <class CharT>
void test() {
  using filebuf = std::basic_filebuf<CharT>;

  { // seek with file closed
    filebuf fb;
    assert(fb.pubseekoff(0, std::ios::beg) == -1);
  }

  {   // seek stream from begin
    { // unbuffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.sgetc() == '1');

      reopen(fb);
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.sgetc() == '2');

      reopen(fb);
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));

      assert(fb.sgetc() == '1'); // prime the buffer

      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.sgetc() == CharT('1'));
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.sgetc() == CharT('2'));
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered, tiny buffer
      filebuf fb;
      CharT buffer[5];
      fb.pubsetbuf(buffer, 5);
      assert(fb.open("seekoff.dat", std::ios::in));
      assert(fb.sgetc() == '1');
      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.sgetc() == CharT('1'));
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.sgetc() == CharT('2'));
      assert(fb.pubseekoff(8, std::ios::beg) == 8);
      assert(fb.sgetc() == CharT('9'));
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // input/output stream, neither read from nor written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));
      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
    }
    { // input/output stream, written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out | std::ios::trunc));
      fb.sputn(MAKE_CSTRING(CharT, "1234567890"), 10);
      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
    }
    { // input/output stream, read from
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));

      assert(fb.pubseekoff(0, std::ios::beg) == 0); // Go to the start, so we can read something

      CharT buffer[11];
      buffer[10] = '\0';
      assert(fb.sgetn(buffer, 10) == 10);
      assert(buffer == MAKE_STRING(CharT, "1234567890"));
      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::beg) == -1);
      assert(fb.pubseekoff(-2, std::ios::beg) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::beg) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::beg) == 0);
      assert(fb.pubseekoff(1, std::ios::beg) == 1);
      assert(fb.pubseekoff(10, std::ios::beg) == 10);
    }
  }
  {   // seek stream from current position
    { // unbuffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      // negative offsets, from begin
      assert(fb.pubseekoff(-1, std::ios::cur) == -1);
      assert(fb.pubseekoff(-2, std::ios::cur) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);

      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 0);
      assert(fb.pubseekoff(1, std::ios::cur) == 1);

      // negative offset, from 1 character into the stream
      assert(fb.pubseekoff(-1, std::ios::cur) == 0);

      assert(fb.pubseekoff(10, std::ios::cur) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      assert(fb.sgetc() == '1');

      // negative offsets, from begin
      assert(fb.pubseekoff(-1, std::ios::cur) == -1);
      assert(fb.pubseekoff(-2, std::ios::cur) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);

      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 0);
      assert(fb.pubseekoff(1, std::ios::cur) == 1);

      // negative offset, from 1 character into the stream
      assert(fb.pubseekoff(-1, std::ios::cur) == 0);

      assert(fb.pubseekoff(10, std::ios::cur) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered, tiny buffer
      filebuf fb;
      CharT buffer[5];
      fb.pubsetbuf(buffer, 5);
      assert(fb.open("seekoff.dat", std::ios::in));
      assert(fb.sgetc() == '1');

      // negative offsets, from begin
      assert(fb.pubseekoff(-1, std::ios::cur) == -1);
      assert(fb.pubseekoff(-2, std::ios::cur) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);

      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 0);
      assert(fb.pubseekoff(1, std::ios::cur) == 1);

      // negative offset, from 1 character into the stream
      assert(fb.pubseekoff(-1, std::ios::cur) == 0);

      assert(fb.pubseekoff(10, std::ios::cur) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // input/output stream, neither read from nor written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));

      // negative offsets
      assert(fb.pubseekoff(-1, std::ios::cur) == -1);
      assert(fb.pubseekoff(-2, std::ios::cur) == -1);
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);
      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 0);
      assert(fb.pubseekoff(1, std::ios::cur) == 1);

      // negative offset, from 1 character into the stream
      assert(fb.pubseekoff(-1, std::ios::cur) == 0);
      assert(fb.pubseekoff(10, std::ios::cur) == 10);
    }
    { // input/output stream, written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out | std::ios::trunc));
      fb.sputn(MAKE_CSTRING(CharT, "1234567890"), 10);

      // negative offsets; we've written and the cursor is at the end of the file, so these succeed
      assert(fb.pubseekoff(-1, std::ios::cur) == 9);
      assert(fb.pubseekoff(-2, std::ios::cur) == 7);

      // except this one, we didn't write this much
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);

      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 7);
      assert(fb.pubseekoff(1, std::ios::cur) == 8);
      assert(fb.pubseekoff(10, std::ios::cur) == 18);
    }
    { // input/output stream, read from
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));

      assert(fb.pubseekoff(0, std::ios::beg) == 0); // Go to the start, so we can read something

      CharT buffer[11];
      buffer[10] = '\0';
      assert(fb.sgetn(buffer, 10) == 10);
      assert(buffer == MAKE_STRING(CharT, "1234567890"));

      // negative offsets; we've read and the cursor is at the end of the file, so these succeed
      assert(fb.pubseekoff(-1, std::ios::cur) == 9);
      assert(fb.pubseekoff(-2, std::ios::cur) == 7);

      // except this one, we didn't read this much
      assert(fb.pubseekoff(std::numeric_limits<typename filebuf::off_type>::min(), std::ios::cur) == -1);

      // zero offset
      assert(fb.pubseekoff(0, std::ios::cur) == 7);
      assert(fb.pubseekoff(1, std::ios::cur) == 8);
      assert(fb.pubseekoff(10, std::ios::cur) == 18);
    }
  }
  {   // seek stream end
    { // unbuffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      assert(fb.sgetc() == '1');
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // buffered, tiny buffer
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in));
      CharT buffer[5];
      fb.pubsetbuf(buffer, 5);
      assert(fb.sgetc() == '1');
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // input/output stream, neither read from nor written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // input/output stream, written to
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out | std::ios::trunc));
      fb.sputn(MAKE_CSTRING(CharT, "1234567890"), 10);
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
    { // input/output stream, read from
      filebuf fb;
      assert(fb.open("seekoff.dat", std::ios::in | std::ios::out));

      assert(fb.pubseekoff(0, std::ios::beg) == 0); // Go to the start, so we can read something

      CharT buffer[11];
      buffer[10] = '\0';
      assert(fb.sgetn(buffer, 10) == 10);
      assert(buffer == MAKE_STRING(CharT, "1234567890"));
      assert(fb.pubseekoff(0, std::ios::end) == 10);
      assert(fb.sgetc() == std::char_traits<CharT>::eof());
    }
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  // TODO: test with different codecvt facets (e.g. where always_noconv() is false, encoding() == 0/-1)

  return 0;
}
