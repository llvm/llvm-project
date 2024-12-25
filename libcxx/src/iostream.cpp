//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_stream.h"
#include <__locale>

_LIBCPP_BEGIN_NAMESPACE_STD

static mbstate_t mb_cin;
static mbstate_t mb_cout;
static mbstate_t mb_cerr;

#if _LIBCPP_HAS_WIDE_CHARACTERS
static mbstate_t mb_wcin;
static mbstate_t mb_wcout;
static mbstate_t mb_wcerr;
#endif

#if __has_cpp_attribute(clang::no_destroy)
template <class T>
using no_destroy_t = T;

#  define STREAM_ATTRS [[clang::no_destroy]] _LIBCPP_INIT_PRIORITY_MAX
#else
template <class T>
using no_destroy_t = __no_destroy<T>;

#  define STREAM_ATTRS _LIBCPP_INIT_PRIORITY_MAX
#endif

auto&& unwrap(auto& val) { return val; }

template <class T>
auto&& unwrap(__no_destroy<T>& val) {
  return val.__get();
}

# 43 __FILE__ 3 // FIXME: Remove this (and the one below) once https://llvm.org/PR121108 is fixed.

STREAM_ATTRS static no_destroy_t<__stdinbuf<char>> cin_stdbuf(stdin, &mb_cin);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<istream> cin(&unwrap(cin_stdbuf));

STREAM_ATTRS static no_destroy_t<__stdoutbuf<char>> cout_stdbuf(stdout, &mb_cout);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<ostream> cout(&unwrap(cout_stdbuf));

STREAM_ATTRS static no_destroy_t<__stdoutbuf<char>> cerr_stdbuf(stderr, &mb_cerr);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<ostream> cerr(&unwrap(cerr_stdbuf));

STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<ostream> clog(&unwrap(cerr_stdbuf));

#if _LIBCPP_HAS_WIDE_CHARACTERS
STREAM_ATTRS static __stdinbuf<wchar_t> wcin_stdbuf(stdin, &mb_wcin);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI wistream wcin(&unwrap(wcin_stdbuf));

STREAM_ATTRS static no_destroy_t<__stdoutbuf<wchar_t>> wcout_stdbuf(stdout, &mb_wcout);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<wostream> wcout(&unwrap(wcout_stdbuf));

STREAM_ATTRS static no_destroy_t<__stdoutbuf<wchar_t>> wcerr_stdbuf(stderr, &mb_wcerr);
STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<wostream> wcerr(&unwrap(wcerr_stdbuf));

STREAM_ATTRS _LIBCPP_EXPORTED_FROM_ABI no_destroy_t<wostream> wclog(&unwrap(wcerr_stdbuf));

#endif // _LIBCPP_HAS_WIDE_CHARACTERS

# 70 __FILE__ 1

// Pretend we're inside a system header so the compiler doesn't flag the use of the init_priority
// attribute with a value that's reserved for the implementation (we're the implementation).
#include "iostream_init.h"

// On Windows the TLS storage for locales needs to be initialized before we create
// the standard streams, otherwise it may not be alive during program termination
// when we flush the streams.
static void force_locale_initialization() {
#if defined(_LIBCPP_MSVCRT_LIKE)
  static bool once = []() {
    auto loc = __locale::__newlocale(LC_ALL_MASK, "C", 0);
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

  unwrap(cin).tie(&unwrap(cout));
  std::unitbuf(unwrap(cerr));
  unwrap(cerr).tie(&unwrap(cout));

#if _LIBCPP_HAS_WIDE_CHARACTERS
  unwrap(wcin).tie(&unwrap(wcout));
  std::unitbuf(unwrap(wcerr));
  unwrap(wcerr).tie(&unwrap(wcout));
#endif
}

DoIOSInit::~DoIOSInit() {
  unwrap(cout).flush();
  unwrap(clog).flush();

#if _LIBCPP_HAS_WIDE_CHARACTERS
  unwrap(wcout).flush();
  unwrap(wclog).flush();
#endif
}

ios_base::Init::Init() {
  static DoIOSInit init_the_streams; // gets initialized once
}

ios_base::Init::~Init() {}

_LIBCPP_END_NAMESPACE_STD
