//===--- msvc_exception_record_fuzzer.cpp - MSVC EH record fuzzer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Feeds synthesized (possibly hostile) MSVC C++ EH EXCEPTION_RECORDs to the
// msvc_exception_ptr domain: type info table walks, the gperf name matcher,
// the what() dispatch and the system_error prefix reads all run against
// fuzz-controlled structures laid out inside a fixed buffer.
//
// Real C++ exception objects are placement-new'ed into the object slot, so
// every path that reaches what() or the error_code prefix stays memory safe;
// the fuzz input controls the EH plumbing (entry count, RVAs, mangled names,
// this-adjustor offsets) that decides which paths get taken.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && defined(_WIN32)

#include "fuzz_harness.h"

#include <windows.h>
#undef min
#undef max

#include <cstring>
#include <new>
#include <stdexcept>
#include <system_error>

namespace {

// Mirrors of the runtime's private EH ABI views
// (src/msvc_exception_ptr.cpp). Keep in sync with those definitions.
struct msvc_cxx_exception_type {
  ::std::uint_least32_t flags;
  ::std::uint_least32_t destructor;
  ::std::uint_least32_t custom_handler;
  ::std::uint_least32_t type_info_table;
};
struct this_ptr_offsets {
  ::std::int_least32_t this_offset;
  ::std::int_least32_t vbase_descr;
  ::std::int_least32_t vbase_offset;
};
struct cxx_type_info {
  ::std::uint_least32_t flags;
  ::std::uint_least32_t type_info; // RVA to std::type_info
  this_ptr_offsets offsets;
  ::std::uint_least32_t size;
  ::std::uint_least32_t copy_ctor;
};
struct cxx_type_info_table {
  ::std::uint_least32_t count;
  ::std::uint_least32_t info[10];
};
struct msvc_raw_type_info {
  void *vtable;
  char *name;
  char mangled[128];
};

constexpr ::std::size_t kBufSize{1u << 16u};
constexpr ::std::uintptr_t kEtOff{0x1000};
constexpr ::std::uintptr_t kTblOff{0x2000};
constexpr ::std::uintptr_t kCtiOff{0x3000};
constexpr ::std::uintptr_t kCtiStride{0x80};
constexpr ::std::uintptr_t kRawOff{0x4000};
constexpr ::std::uintptr_t kRawStride{0x100};
constexpr ::std::uintptr_t kObjOff{0x6000};
constexpr unsigned kMaxEntries{5};

alignas(16) char unsigned g_buf[kBufSize];

void place_object(::std::uint8_t selector) {
  switch (selector & 3u) {
  case 0:
    ::new (g_buf + kObjOff)::std::runtime_error("fuzz");
    break;
  case 1:
    ::new (g_buf + kObjOff)::std::logic_error("fuzz");
    break;
  case 2:
    ::new (g_buf + kObjOff)::std::system_error(
        ::std::error_code(EACCES, ::std::generic_category()));
    break;
  default:
    ::new (g_buf + kObjOff)::std::out_of_range("fuzz");
    break;
  }
}

void copy_mangled(msvc_raw_type_info *raw, ::std::uint8_t const *data,
                  ::std::size_t size) noexcept {
  ::std::size_t const n{size < 127 ? size : 127};
  for (::std::size_t i = 0; i != n; ++i) {
    raw->mangled[i] = static_cast<char>(data[i]);
  }
  raw->mangled[n] = '\0';
}

void seed_mangled(msvc_raw_type_info *raw, char const *name) noexcept {
  unsigned i = 0;
  for (; name[i] != '\0'; ++i) {
    raw->mangled[i] = name[i];
  }
  raw->mangled[i] = '\0';
}

void run_queries(::std::error_domain_singleton const *domain, void *storage) {
  herbceptions_fuzz::capture_ctx ctx;
  auto const cd{reinterpret_cast<::std::size_t>(&storage)};
  constexpr ::std::error_query_information kQueries[] = {
      ::std::error_query_information::name,
      ::std::error_query_information::message,
      ::std::error_query_information::name_message,
  };
  constexpr ::std::error_reporter_encoding kEncodings[] = {
      ::std::error_reporter_encoding::utf8,    ::std::error_reporter_encoding::utf16,
      ::std::error_reporter_encoding::utf32,   ::std::error_reporter_encoding::gb18030,
      ::std::error_reporter_encoding::utfebcdic,
  };
  for (auto const q : kQueries) {
    for (auto const e : kEncodings) {
      herbceptions_fuzz::reset(ctx);
      domain->do_query_information(cd, q, e, &ctx,
                                   herbceptions_fuzz::capture_cookfun);
    }
  }
  domain->do_to_errc(cd);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *data,
                                      ::std::size_t size) {
  if (size < 16 || data[0] != 'H' || data[1] != 'E') {
    return 0;
  }

  for (::std::size_t i = 0; i != kBufSize; ++i) {
    g_buf[i] = 0;
  }
  place_object(data[3]);

  auto *et{reinterpret_cast<msvc_cxx_exception_type *>(g_buf + kEtOff)};
  auto *tbl{reinterpret_cast<cxx_type_info_table *>(g_buf + kTblOff)};

  unsigned const entries{static_cast<unsigned>(data[2] % kMaxEntries) + 1u};
  tbl->count = entries;

  et->flags = 0;
  et->destructor = 0;
  et->custom_handler = 0;
  et->type_info_table = static_cast<::std::uint_least32_t>(kTblOff);

  // The exact mangled names the runtime searches for, seeded on demand so
  // the what() dispatch and the gperf/system_error paths stay reachable.
  char const kWhatName[]{".?AVexception@stdext@@"};
  char const kSysErrName[]{".?AV_System_error@std@@"};
  bool const seed_what{(data[5] & 1u) != 0};
  bool const seed_syserr{(data[5] & 2u) != 0};

  for (unsigned i = 0; i < entries; ++i) {
    auto *cti{
        reinterpret_cast<cxx_type_info *>(g_buf + kCtiOff + i * kCtiStride)};
    cti->flags = 0;
    cti->offsets.this_offset =
        static_cast<::std::int_least32_t>(static_cast<signed char>(data[9])) *
        4;
    cti->offsets.vbase_descr = -1;
    cti->offsets.vbase_offset = 0;
    cti->size = 512;
    cti->copy_ctor = 0;
    // Bounded jitter keeps the RVA inside the raw type info region while
    // still letting the fuzzer explore misalignment / wrong-slot choices.
    ::std::uintptr_t const target{kRawOff + i * kRawStride +
                                  static_cast<::std::uintptr_t>(data[4] % 4u) *
                                      32u};
    cti->type_info = static_cast<::std::uint_least32_t>(target);

    auto *raw{reinterpret_cast<msvc_raw_type_info *>(g_buf + target)};
    raw->vtable = nullptr;
    raw->name = nullptr;
    ::std::size_t const payload_off{10 + i * 8};
    if (seed_what && i == data[6] % entries) {
      seed_mangled(raw, kWhatName);
    } else if (seed_syserr && i == data[7] % entries) {
      seed_mangled(raw, kSysErrName);
    } else if (payload_off < size) {
      copy_mangled(raw, data + payload_off, size - payload_off);
    }
  }

  EXCEPTION_RECORD rec{};
  rec.ExceptionCode = 0xe06d7363; // MSVC C++ EH
  rec.NumberParameters = 4;
  rec.ExceptionInformation[0] = 0x19930520;
  rec.ExceptionInformation[1] =
      reinterpret_cast<::std::uintptr_t>(g_buf + kObjOff);
  rec.ExceptionInformation[2] =
      reinterpret_cast<::std::uintptr_t>(g_buf + kEtOff);
  rec.ExceptionInformation[3] = 0;

  void *storage{&rec};
  run_queries(::std::error_domains::__cxa_error_domain_msvc_exception_ptr(),
              storage);
  return 0;
}

#else

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *,
                                      ::std::size_t) {
  return 0;
}

#endif
