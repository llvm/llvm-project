//===--- itanium_exception_record_fuzzer.cpp - itanium EH fuzzer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Feeds synthesized __cxa_exception headers to the itanium_exception_ptr
// domain. The exception type is always one of a fixed set of REAL type_info
// objects and the thrown object slot holds a REAL live exception, so every
// path that reaches what() or the RTTI walk stays memory safe while the
// fuzz input decides which paths get exercised.
//
//===----------------------------------------------------------------------===//

#if !defined(_MSC_VER)

#include "fuzz_harness.h"

#include "../src/itanium_exception_ptr.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <system_error>

namespace {

constexpr ::std::size_t kBufSize{1u << 16u};

alignas(16) char unsigned g_buf[kBufSize];

void place_object(::std::uint8_t selector, void *obj) {
  switch (selector & 3u) {
  case 0:
    ::new (obj)::std::runtime_error("fuzz");
    break;
  case 1:
    ::new (obj)::std::logic_error("fuzz");
    break;
  case 2:
    ::new (obj)::std::system_error(
        ::std::error_code(EACCES, ::std::generic_category()));
    break;
  default:
    ::new (obj)::std::out_of_range("fuzz");
    break;
  }
}

::std::type_info const &type_for(::std::uint8_t selector) {
  switch (selector & 3u) {
  case 0:
    return typeid(::std::runtime_error);
  case 1:
    return typeid(::std::logic_error);
  case 2:
    return typeid(::std::system_error);
  default:
    return typeid(::std::out_of_range);
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *data,
                                      ::std::size_t size) {
  if (size < 8 || data[0] != 'I' || data[1] != 'T') {
    return 0;
  }

  for (::std::size_t i = 0; i != kBufSize; ++i) {
    g_buf[i] = 0;
  }

  ::std::size_t const thrown_off{
      (sizeof(itanium_cxa_exception) + 15u) & ~::std::size_t{15u}};
  auto *hdr{reinterpret_cast<itanium_cxa_exception *>(g_buf)};
  void *const obj{g_buf + thrown_off};
  place_object(data[3], obj);

  hdr->reserve = nullptr;
  hdr->referenceCount = 1;
  hdr->exceptionType =
      const_cast<::std::type_info *>(&type_for(data[2]));
  hdr->exceptionDestructor = nullptr;
  hdr->unexpectedHandler = nullptr;
  hdr->terminateHandler = nullptr;
  hdr->nextException = nullptr;
  hdr->handlerCount = 0;
  hdr->handlerSwitchValue = 0;
  hdr->actionRecord = nullptr;
  hdr->languageSpecificData = nullptr;
  hdr->catchTemp = nullptr;
  hdr->adjustedPtr = nullptr;
  hdr->unwindHeader.exception_class = 0x002B2B4343554E47ull; // "GNUCC++\\0"

  herbceptions_fuzz::capture_ctx ctx;
  auto const cd{reinterpret_cast<::std::size_t>(obj)};
  constexpr ::std::error_query_information kQueries[] = {
      ::std::error_query_information::name,
      ::std::error_query_information::message,
      ::std::error_query_information::name_message,
  };
  constexpr ::std::error_reporter_encoding kEncodings[] = {
      ::std::error_reporter_encoding::utf8,
      ::std::error_reporter_encoding::utf16,
      ::std::error_reporter_encoding::utf32,
      ::std::error_reporter_encoding::gb18030,
      ::std::error_reporter_encoding::utfebcdic,
  };
  auto const *domain{
      ::std::error_domains::__cxa_error_domain_itanium_exception_ptr()};
  for (auto const q : kQueries) {
    for (auto const e : kEncodings) {
      herbceptions_fuzz::reset(ctx);
      domain->do_query_information(cd, q, e, &ctx,
                                   herbceptions_fuzz::capture_cookfun);
    }
  }
  domain->do_to_errc(cd);
  return 0;
}

#else

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *,
                                      ::std::size_t) {
  return 0;
}

#endif
