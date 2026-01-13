//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test requires the fix to std::filebuf::close() (which is defined in the
// built library) from https://github.com/llvm/llvm-project/pull/168947.
// UNSUPPORTED: using-built-library-before-llvm-22

// UNSUPPORTED: no-localization, no-filesystem

// <fstream>

// basic_filebuf<charT,traits>* close();

//
// Ensure that basic_filebuf::close() does not get rid of the underlying buffer set
// via pubsetbuf(). Otherwise, reopening the stream will result in not reusing the
// same buffer, which is conforming but also very surprising.
//

#include <cassert>
#include <fstream>
#include <string>

#include "platform_support.h"
#include "test_macros.h"

struct overflow_detecting_filebuf : std::filebuf {
  explicit overflow_detecting_filebuf(bool* overflow_monitor) : did_overflow_(overflow_monitor) {
    assert(overflow_monitor != nullptr && "must provide an overflow monitor");
  }

  using Traits = std::filebuf::traits_type;
  virtual std::filebuf::int_type overflow(std::filebuf::int_type ch = Traits::eof()) {
    *did_overflow_ = true;
    return std::filebuf::overflow(ch);
  }

private:
  bool* did_overflow_;
};

int main(int, char**) {
  std::string temp = get_temp_file_name();

  bool did_overflow;
  overflow_detecting_filebuf buf(&did_overflow);

  // Set a custom buffer (of size 32, reused below)
  char underlying_buffer[32];
  buf.pubsetbuf(underlying_buffer, sizeof(underlying_buffer));

  // (1) Open a file and insert a first character. That should overflow() and set the underlying
  //     put area to our internal buffer set above.
  {
    buf.open(temp, std::ios::out | std::ios::trunc);
    did_overflow = false;
    buf.sputc('c');
    assert(did_overflow == true);
  }

  // (2) Now, confirm that we can still insert 30 more characters without calling
  //     overflow, since we should be writing to the internal buffer.
  {
    did_overflow = false;
    for (int i = 0; i != 30; ++i) {
      buf.sputc('c');
      assert(did_overflow == false);
    }
  }

  // (3) Writing the last character may or may not call overflow(), depending on whether
  //     the library implementation wants to flush as soon as the underlying buffer is
  //     full, or on the next attempt to insert. For libc++, it doesn't overflow yet.
  {
    did_overflow = false;
    buf.sputc('c');
    LIBCPP_ASSERT(!did_overflow);
  }

  // (4) Writing one-too-many characters will overflow (with libc++).
  {
    did_overflow = false;
    buf.sputc('c');
    LIBCPP_ASSERT(did_overflow);
  }

  // Close the stream. This should NOT unset the underlying buffer we set at the beginning
  // Unfortunately, the only way to check that is to repeat the above tests which tries to
  // tie the presence of our custom set buffer to whether overflow() gets called. This is
  // not entirely portable since implementations are free to call overflow() whenever they
  // want, but in practice this works pretty portably.
  buf.close();

  // Repeat (1)
  {
    buf.open(temp, std::ios::out | std::ios::trunc);
    did_overflow = false;
    buf.sputc('c');
    assert(did_overflow == true);
  }

  // Repeat (2)
  {
    did_overflow = false;
    for (int i = 0; i != 30; ++i) {
      buf.sputc('c');
      assert(did_overflow == false);
    }
  }

  // Repeat (3)
  {
    did_overflow = false;
    buf.sputc('c');
    LIBCPP_ASSERT(!did_overflow);
  }

  // Repeat (4)
  {
    did_overflow = false;
    buf.sputc('c');
    LIBCPP_ASSERT(did_overflow);
  }

  return 0;
}
