//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_stream.h"

#include <__memory/construct_at.h>
#include <__ostream/basic_ostream.h>
#include <istream>

#define ABI_NAMESPACE_STR _LIBCPP_TOSTRING(_LIBCPP_ABI_NAMESPACE)

_LIBCPP_BEGIN_NAMESPACE_STD

// This file implements the various stream objects provided inside <iostream>. We're doing some ODR violations in here,
// so this quite fragile. Specifically, the size of the stream objects (i.e. cout, cin etc.) needs to stay the same.
// For that reason, we have `stream` and `stream_data` separated into two objects. The public `stream` objects only
// contain the actual stream, while the private `stream_data` objects contains the `basic_streambuf` we're using as well
// as the mbstate_t. `stream_data` objects are only accessible within the library, so they aren't ABI sensitive and we
// can change them as we want.

template <class StreamT>
union stream {
  constexpr stream() {}
  stream(const stream&)            = delete;
  stream& operator=(const stream&) = delete;
  constexpr ~stream() {}

  StreamT value;
};

template <class StreamT, class BufferT>
union stream_data {
  constexpr stream_data() {}
  constexpr ~stream_data() {}
  struct {
    BufferT buffer;
    mbstate_t mb;
  };
};

template <class StreamT, class BufferT>
void init_stream(FILE* stdstream, stream<StreamT>& stream, stream_data<StreamT, BufferT>& data) {
  data.mb = {};
  std::construct_at(&data.buffer, stdstream, &data.mb);
  std::construct_at(&stream.value, &data.buffer);
}

#define CHAR_MANGLING_char "D"
#define CHAR_MANGLING_wchar_t "_W"
#define CHAR_MANGLING(CharT) CHAR_MANGLING_##CharT

#ifdef _LIBCPP_COMPILER_CLANG_BASED
#  define STRING_DATA_CONSTINIT constinit
#else
#  define STRING_DATA_CONSTINIT
#endif

#ifdef _LIBCPP_ABI_MICROSOFT
#  define STREAM(StreamT, BufferT, CharT, var)                                                                         \
    STRING_DATA_CONSTINIT stream_data<StreamT<CharT>, BufferT<CharT>> var##_data;                                      \
    _LIBCPP_EXPORTED_FROM_ABI STRING_DATA_CONSTINIT stream<StreamT<CharT>> var __asm__(                                \
        "?" #var "@" ABI_NAMESPACE_STR "@std@@3V?$" #StreamT                                                           \
        "@" CHAR_MANGLING(CharT) "U?$char_traits@" CHAR_MANGLING(CharT) "@" ABI_NAMESPACE_STR "@std@@@12@A")
#else
#  define STREAM(StreamT, BufferT, CharT, var)                                                                         \
    STRING_DATA_CONSTINIT stream_data<StreamT<CharT>, BufferT<CharT>> var##_data;                                      \
    _LIBCPP_EXPORTED_FROM_ABI STRING_DATA_CONSTINIT stream<StreamT<CharT>> var
#endif

// These definitions and the declarations in <iostream> technically cause ODR violations, since they have different
// types (stream_data and {i,o}stream respectively). This means that <iostream> should never be included in this TU.

STREAM(basic_istream, __stdinbuf, char, cin);
STREAM(basic_ostream, __stdoutbuf, char, cout);
STREAM(basic_ostream, __stdoutbuf, char, cerr);
STREAM(basic_ostream, __stdoutbuf, char, clog);
#if _LIBCPP_HAS_WIDE_CHARACTERS
STREAM(basic_istream, __stdinbuf, wchar_t, wcin);
STREAM(basic_ostream, __stdoutbuf, wchar_t, wcout);
STREAM(basic_ostream, __stdoutbuf, wchar_t, wcerr);
STREAM(basic_ostream, __stdoutbuf, wchar_t, wclog);
#endif // _LIBCPP_HAS_WIDE_CHARACTERS

// Pretend we're inside a system header so the compiler doesn't flag the use of the init_priority
// attribute with a value that's reserved for the implementation (we're the implementation).
#include "iostream_init.h"

// On Windows the TLS storage for locales needs to be initialized before we create
// the standard streams, otherwise it may not be alive during program termination
// when we flush the streams.
static void force_locale_initialization() {
#if defined(_LIBCPP_MSVCRT_LIKE)
  static bool once = []() {
    auto loc = __locale::__newlocale(_LIBCPP_ALL_MASK, "C", 0);
    {
      __locale::__locale_guard g(loc); // forces initialization of locale TLS
      ((void)g);
    }
    __locale::__freelocale(loc);
    return true;
  }();
  ((void)once);
#endif
}

class DoIOSInit {
public:
  DoIOSInit();
  ~DoIOSInit();
};

DoIOSInit::DoIOSInit() {
  force_locale_initialization();

  init_stream(stdin, cin, cin_data);
  init_stream(stdout, cout, cout_data);
  init_stream(stderr, cerr, cerr_data);
  init_stream(stderr, clog, clog_data);

  cin.value.tie(&cout.value);
  std::unitbuf(cerr.value);
  cerr.value.tie(&cout.value);

#if _LIBCPP_HAS_WIDE_CHARACTERS
  init_stream(stdin, wcin, wcin_data);
  init_stream(stdout, wcout, wcout_data);
  init_stream(stderr, wcerr, wcerr_data);
  init_stream(stderr, wclog, wclog_data);

  wcin.value.tie(&wcout.value);
  std::unitbuf(wcerr.value);
  wcerr.value.tie(&wcout.value);
#endif
}

DoIOSInit::~DoIOSInit() {
  cout.value.flush();
  clog.value.flush();

#if _LIBCPP_HAS_WIDE_CHARACTERS
  wcout.value.flush();
  wclog.value.flush();
#endif
}

ios_base::Init::Init() {
  static DoIOSInit init_the_streams; // gets initialized once
}

ios_base::Init::~Init() {}

_LIBCPP_END_NAMESPACE_STD
