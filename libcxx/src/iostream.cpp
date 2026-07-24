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
#include <fstream>
#include <istream>

#define ABI_NAMESPACE_STR _LIBCPP_TOSTRING(_LIBCPP_ABI_NAMESPACE)

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

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

template <class StreamT, class SyncBufT, class UnsyncBufT>
union stream_data {
  constexpr stream_data() {}
  constexpr ~stream_data() {}
  struct {
    union {
      SyncBufT sync_buffer;
      UnsyncBufT unsync_buffer;
    };
    mbstate_t mb;
  };
};

template <class StreamT, class SyncBufT, class UnsyncBufT>
void init_stream(FILE* stdstream, stream<StreamT>& stream, stream_data<StreamT, SyncBufT, UnsyncBufT>& data) {
  data.mb = {};
  std::construct_at(&data.sync_buffer, stdstream, &data.mb);
  std::construct_at(&stream.value, &data.sync_buffer);
}

template <class StreamT, class SyncBufT, class UnsyncBufT>
void switch_to_sync_stream(FILE* stdstream, stream<StreamT>& stream, stream_data<StreamT, SyncBufT, UnsyncBufT>& data) {
  data.unsync_buffer.__adopt_file(nullptr, {}); // reset the file, so that basic_filebuf doesn't close standard streams
  std::destroy_at(&data.unsync_buffer);
  data.mb = {};
  std::construct_at(&data.sync_buffer, stdstream, &data.mb);
  stream.value.rdbuf(&data.sync_buffer);
}

template <class StreamT, class SyncBufT, class UnsyncBufT>
void switch_to_unsync_stream(FILE* stdstream,
                             stream<StreamT>& stream,
                             stream_data<StreamT, SyncBufT, UnsyncBufT>& data,
                             ios_base::openmode mode) {
  std::destroy_at(&data.sync_buffer);
  std::construct_at(&data.unsync_buffer);
  data.unsync_buffer.__adopt_file(stdstream, mode);
  stream.value.rdbuf(&data.unsync_buffer);
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
#  define STREAM(StreamT, SyncBufT, UnsyncBufT, CharT, var)                                                            \
    STRING_DATA_CONSTINIT stream_data<StreamT<CharT>, SyncBufT<CharT>, UnsyncBufT<CharT>> var##_data;                  \
    _LIBCPP_EXPORTED_FROM_ABI STRING_DATA_CONSTINIT stream<StreamT<CharT>> var __asm__(                                \
        "?" #var "@" ABI_NAMESPACE_STR "@std@@3V?$" #StreamT                                                           \
        "@" CHAR_MANGLING(CharT) "U?$char_traits@" CHAR_MANGLING(CharT) "@" ABI_NAMESPACE_STR "@std@@@12@A")
#else
#  define STREAM(StreamT, SyncBufT, UnsyncBufT, CharT, var)                                                            \
    STRING_DATA_CONSTINIT stream_data<StreamT<CharT>, SyncBufT<CharT>, UnsyncBufT<CharT>> var##_data;                  \
    _LIBCPP_EXPORTED_FROM_ABI STRING_DATA_CONSTINIT stream<StreamT<CharT>> var
#endif

// These definitions and the declarations in <iostream> technically cause ODR violations, since they have different
// types (stream_data and {i,o}stream respectively). This means that <iostream> should never be included in this TU.

#if _LIBCPP_HAS_FILESYSTEM
template <class CharT>
using unsync_buffer = basic_filebuf<CharT>;
#else
// If we don't have access to files we can't treat the standard streams as files either - just save a dummy object in
// that case.
template <class>
using unsync_buffer = char;
#endif

STREAM(basic_istream, __stdinbuf, unsync_buffer, char, cin);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, char, cout);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, char, cerr);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, char, clog);
#if _LIBCPP_HAS_WIDE_CHARACTERS
STREAM(basic_istream, __stdinbuf, unsync_buffer, wchar_t, wcin);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, wchar_t, wcout);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, wchar_t, wcerr);
STREAM(basic_ostream, __stdoutbuf, unsync_buffer, wchar_t, wclog);
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

bool ios_base::sync_with_stdio(bool sync) {
  static bool previous_state = true;
  bool r                     = previous_state;

#if _LIBCPP_HAS_FILESYSTEM
  if (sync != previous_state) {
    if (sync) {
      switch_to_sync_stream(stdin, cin, cin_data);
      switch_to_sync_stream(stdout, cout, cout_data);
      switch_to_sync_stream(stderr, clog, clog_data);
#  if _LIBCPP_HAS_WIDE_CHARACTERS
      switch_to_sync_stream(stdin, wcin, wcin_data);
      switch_to_sync_stream(stdout, wcout, wcout_data);
      switch_to_sync_stream(stderr, wclog, wclog_data);
#  endif
    } else {
      switch_to_unsync_stream(stdin, cin, cin_data, ios::in);
      switch_to_unsync_stream(stdout, cout, cout_data, ios::out);
      switch_to_unsync_stream(stderr, clog, clog_data, ios::out);
#  if _LIBCPP_HAS_WIDE_CHARACTERS
      switch_to_unsync_stream(stdin, wcin, wcin_data, ios::in);
      switch_to_unsync_stream(stdout, wcout, wcout_data, ios::out);
      switch_to_unsync_stream(stderr, wclog, wclog_data, ios::out);
#  endif
    }
  }
#endif // _LIBCPP_HAS_FILESYSTEM

  previous_state = sync;
  return r;
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
