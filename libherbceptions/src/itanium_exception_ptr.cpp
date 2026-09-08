//===--- itanium_exception_ptr.cpp - itanium exception_ptr domain --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the itanium exception_ptr error_domain_singleton vtable and the
// weak __cxa_error_domain_itanium_exception_ptr ABI entry point. Available on
// all platforms. The code is the thrown-object pointer (the __cxa catch
// value); the vtable releases it on cleanup and rethrows it via the ABI.
//
// The __cxa_* ABI (allocate/free/refcount/rethrow) exists only on Itanium-ABI
// targets (Linux, macOS, Cygwin, MinGW). The MSVC ABI (indicated by _MSC_VER)
// uses __CxxFrameHandler3/_CxxThrowException instead, so the vtable degrades
// gracefully there (no refcount/rethrow support).
//
//===----------------------------------------------------------------------===//

#if !defined(_MSC_VER) && defined(__cpp_exceptions)
#include "itanium_exception_ptr.h"
#include "__malloc_or_heap_alloc_temp_buffer.h"
#include "libherbceptions.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <system_error>
#include <typeinfo>

namespace std::error_domains {
namespace {
using namespace __herbceptions_detail;

// Walk the Itanium RTTI base-class graph WITHOUT touching the thrown
// object. True iff dst is reachable through single-inheritance (non-virtual)
// links only. Such primary-chain bases all sit at offset 0 within the
// complete object, so a successful answer both locates the std::exception
// subobject at the thrown-object address and guarantees polymorphism
// (std::exception is polymorphic), making virtual dispatch safe on throws
// of any type -- including ints and plain structs, which fail the walk.
//
// Shape discrimination compares the type_info's own vptr against reference
// RTTI objects from this TU: typeid(std::exception) has no bases (plain
// __class_type_info), typeid(std::logic_error) has exactly one base
// (__si_class_type_info). Multiple/virtual-inheritance (__vmi) and foreign
// shapes are rejected conservatively -> name-only reporting. (GCC's
// __dynamic_cast cannot be used here: its contract is downcast-oriented,
// and passing the dynamic type as src_type yields NULL.)
inline void const *ti_vptr(::std::type_info const &ti) noexcept {
  return *static_cast<void const *const *>(static_cast<void const *>(&ti));
}

inline bool rtti_si_derives_from(::std::type_info const *ti,
                                 ::std::type_info const *dst) noexcept {
  if (!ti || !dst) {
    return false;
  }
  void const *const si_vtbl{ti_vptr(typeid(::std::logic_error))};
  auto cur{static_cast<::__cxxabiv1::__class_type_info const *>(ti)};
  for (;;) {
    if (*static_cast<::std::type_info const *>(cur) == *dst) {
      return true;
    }
    void const *vp{ti_vptr(*static_cast<::std::type_info const *>(cur))};
    if (vp == si_vtbl) {
      cur = reinterpret_cast<::__cxxabiv1::__si_class_type_info const *>(cur)
                ->__base_type;
      continue;
    }
    // Plain root / fundamental / pointer / VMI / foreign: stop.
    return false;
  }
}

// Encoding-specific wrapper fragments carved out of the base literal
// "[itanium_exception](?)" (22 code units), mirroring the MSVC sibling:
//   startpos 0 len 22 -> "[itanium_exception](?)"
//   startpos 0 len 20 -> "[itanium_exception](" (known prefix)
//   startpos 0 len 19 -> "[itanium_exception]"
//   startpos 1 len 17 -> "itanium_exception"
//   startpos 19 len 3 -> "(?)"  /  len 1 -> "("
//   startpos 21 len 1 -> ")"
inline constexpr ::std::io_scatter_t
itanium_exception_name_message_range(::std::error_reporter_encoding encoding,
                                     ::std::size_t startpos,
                                     ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    // EBCDIC bytes per __ascii_to_ebcdic.
    return {&startpos["\xBA\x89\xA3\x81\x95\x89\xA4\x94\x6D\x85\xA7\x83\x85"
                      "\x97\xA3\x89\x96\x95\x5A\x4D\x6F\x5D"],
            n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[itanium_exception](?)"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[itanium_exception](?)"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[itanium_exception](?)"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
itanium_exception_name(::std::error_reporter_encoding encoding) noexcept {
  /*
  itanium_exception
  */
  return itanium_exception_name_message_range(encoding, 1, 17u);
}

inline constexpr ::std::io_scatter_t unknown_itanium_exception_name_message(
    ::std::error_reporter_encoding encoding) noexcept {
  /*
  [itanium_exception](?)
  */
  return itanium_exception_name_message_range(encoding, 0, 22u);
}

inline constexpr ::std::io_scatter_t unknown_itanium_exception_message_partial(
    ::std::error_reporter_encoding encoding) noexcept {
  /*
  (?)
  */
  return itanium_exception_name_message_range(encoding, 19, 3u);
}

inline constexpr ::std::io_scatter_t
known_itanium_exception_name_message_partial(
    ::std::error_reporter_encoding encoding) noexcept {
  /*
  [itanium_exception](
  */
  return itanium_exception_name_message_range(encoding, 0, 20u);
}

inline constexpr ::std::io_scatter_t
left_parenthese(::std::error_reporter_encoding encoding) noexcept {
  /*
  (
  */
  return itanium_exception_name_message_range(encoding, 19, 1u);
}

inline constexpr ::std::io_scatter_t
right_parenthese(::std::error_reporter_encoding encoding) noexcept {
  /*
  )
  */
  return itanium_exception_name_message_range(encoding, 21, 1u);
}

struct itanium_exception_writestr_return {
  ::std::io_scatter_t name;
  ::std::io_scatter_t message;
};

inline itanium_exception_writestr_return itanium_exception_writestr(
    char const *name, ::std::size_t namelen, char const *message,
    ::std::size_t messagelen, ::std::error_reporter_encoding encoding,
    ::std::error_domains::__herbceptions_detail::
        __malloc_or_heapalloc_temp_buffer &buffer) noexcept {
  static_assert('A' == u8'A', "EBCDIC Execution Charset not supported");
  if (
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
      encoding == ::std::error_reporter_encoding::utfebcdic ||
#endif
      encoding == ::std::error_reporter_encoding::utf16 ||
      encoding == ::std::error_reporter_encoding::utf32) {
    ::std::size_t to_allocate_bytes{};
    if (__builtin_add_overflow(namelen, messagelen,
                               __builtin_addressof(to_allocate_bytes))) {
      abort();
    }
    if (to_allocate_bytes) {
      ::std::size_t szch{1u};
      if (encoding == ::std::error_reporter_encoding::utf16) {
        szch = 2u;
      } else if (encoding == ::std::error_reporter_encoding::utf32) {
        szch = 4u;
      }
      if (__builtin_mul_overflow(to_allocate_bytes, szch,
                                 __builtin_addressof(to_allocate_bytes))) {
        abort();
      }
#ifdef _WIN32
      buffer.__bufferptr = ::std::error_domains::__herbceptions_detail::
          __win32_heap_alloc_or_die(to_allocate_bytes);
#else
      buffer.__bufferptr = ::std::error_domains::__herbceptions_detail::
          __malloc_or_heap_alloc_or_die(to_allocate_bytes);
#endif
      char unsigned *bufferptr{
          reinterpret_cast<char unsigned *>(buffer.__bufferptr)};
      auto name_end{
          ::std::error_domains::__herbceptions_detail::
              __codecvt_write_with_encoding(
                  reinterpret_cast<char unsigned const *>(name),
                  reinterpret_cast<char unsigned const *>(name + namelen),
                  bufferptr, encoding)};
      auto message_end{
          ::std::error_domains::__herbceptions_detail::
              __codecvt_write_with_encoding(
                  reinterpret_cast<char unsigned const *>(message),
                  reinterpret_cast<char unsigned const *>(message + messagelen),
                  name_end, encoding)};
      return {{bufferptr, static_cast<::std::size_t>(name_end - bufferptr)},
              {name_end, static_cast<::std::size_t>(message_end - name_end)}};
    }
  }
  return {{name, namelen}, {message, messagelen}};
}

// True iff a thrown object whose dynamic RTTI is exc_ti could be caught
// as Kind -- i.e. the dynamic_cast<Kind> question, answered with the
// runtime's own catch matcher: kind.__do_catch(exc_ti, &obj), the very
// call chain the personality routine uses (get_adjusted_ptr). Public API,
// no internal structures. Class-kind matching walks typeinfo graphs and
// does pointer arithmetic only; pointer-kind catches deref obj once.
#if defined(__GLIBCXX__)
inline bool itanium_cxa_try_catch(::std::type_info const *exc_ti,
                                  void const *obj, ::std::type_info const &kind,
                                  void **out) noexcept {
  if (!exc_ti) {
    return false;
  }
  void const *adjusted{obj};
  if (!kind.__do_catch(
          exc_ti, const_cast<void **>(__builtin_addressof(adjusted)), 0u)) {
    return false;
  }
  // The catch matcher left the Kind subobject address in adjusted.
  *out = const_cast<void *>(adjusted);
  return true;
}
#else
inline bool itanium_cxa_try_catch(::std::type_info const *exc_ti,
                                  void const *obj, ::std::type_info const &kind,
                                  void **out) noexcept {
  // Non-libstdc++ runtimes ship different catch virtuals; restrict to
  // single-inheritance chains, where the base subobject provably sits
  // at offset 0.
  if (!rtti_si_derives_from(exc_ti, __builtin_addressof(kind))) {
    return false;
  }
  *out = const_cast<void *>(obj);
  return true;
}
#endif

inline bool itanium_cxa_catchable(::std::type_info const *exc_ti,
                                  void const *obj,
                                  ::std::type_info const &kind) noexcept {
  void *unused{};
  return itanium_cxa_try_catch(exc_ti, obj, kind, &unused);
}

// Mirror of the runtimes' __cxa_eh_globals (libstdc++ keeps the public type
// opaque); layout stable since the ABI's inception.
struct itanium_cxa_eh_globals {
  void *caughtExceptions;
  unsigned int uncaughtExceptions;
};

constinit ::std::error_domain_singleton itanium_exception_ptr_domain{
    .do_cleanup =
        [](::std::size_t cd) noexcept {
          __itanium_cxa_decrement_exception_refcount(
              reinterpret_cast<void *>(cd));
        },
    .do_equivalent = [](::std::size_t cd,
                        ::std::error_domain_singleton const *other,
                        ::std::size_t othercd) noexcept -> bool {
      return itanium_exception_ptr_domain.do_to_errc(cd) ==
             other->do_to_errc(othercd);
    },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          if (static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message) <
              static_cast<::std::uint_least32_t>(query)) {
            return;
          }
          // cd is the thrown-object pointer (the value stored by
          // error_domain<exception_ptr>::code); the __cxa_exception header
          // sits immediately before it. Foreign EH can never reach here:
          // __cxa_error_code_itanium_exception_ptr aborts on it upstream.
          void *thrown{reinterpret_cast<void *>(cd)};
          auto *hdr{thrown ? itanium_cxa_exception_from_thrown_object(thrown)
                           : nullptr};
          bool const is_itanium_cxx_eh{hdr != nullptr};
          if constexpr (::std::error_domains::__herbceptions_detail::
                            __is_freestanding_kernel_mode) {
            /*
            Freestanding kernel mode: no heap allocation and no big stack
            allocation. The mangled name and what() text are narrow (utf8 /
            gb18030 compatible) bytes that are already available in place,
            so they are only reported for those encodings; every other
            encoding would require a conversion buffer, so it degrades to
            (?) with no message.
            */
            bool const printable_encoding{
                encoding == ::std::error_reporter_encoding::utf8 ||
                encoding == ::std::error_reporter_encoding::gb18030};
            ::std::io_scatter_t scatters[4];
            ::std::size_t scatterlen{};
            char const *ehname{};
            ::std::size_t ehname_len{};
            char const *ehmessage{};
            ::std::size_t ehmessage_len{};
            if (is_itanium_cxx_eh &&
                ::std::error_query_information::name != query) {
              // Raw mangled form; no demangling, no heap allocation.
              if (hdr->exceptionType) {
                ehname = hdr->exceptionType->name();
                ehname_len = ::std::strlen(ehname);
              }
              // what() lives on the std::exception subobject, not on
              // type_info. When the RTTI graph proves a single-inheritance
              // path to std::exception, that subobject sits at offset 0 of
              // the thrown object, so dispatch directly through it.
              if (printable_encoding && thrown &&
                  rtti_si_derives_from(
                      hdr->exceptionType,
                      __builtin_addressof(typeid(::std::exception)))) {
                auto *se{static_cast<::std::exception *>(thrown)};
                ehmessage = se->what();
                if (ehmessage) {
                  ehmessage_len = ::std::strlen(ehmessage);
                }
              }
            }
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = itanium_exception_name(encoding);
              scatterlen = 1u;
              break;
            }
            case ::std::error_query_information::message:
              [[fallthrough]];
            case ::std::error_query_information::name_message: {
              if (::std::error_query_information::name_message == query) {
                if (printable_encoding && ehname_len) {
                  *scatters =
                      known_itanium_exception_name_message_partial(encoding);
                } else {
                  *scatters = unknown_itanium_exception_name_message(encoding);
                }
                ++scatterlen;
              }
              if (printable_encoding && ehname_len) {
                scatters[scatterlen] = left_parenthese(encoding);
                scatters[scatterlen + 1u] = {ehname, ehname_len};
                scatters[scatterlen + 2u] = right_parenthese(encoding);
                scatterlen += 3u;
              } else {
                scatters[scatterlen] =
                    unknown_itanium_exception_message_partial(encoding);
                ++scatterlen;
              }
              if (printable_encoding && ehmessage_len) {
                scatters[scatterlen] = {ehmessage, ehmessage_len};
                ++scatterlen;
              }
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          } else {
            ::std::error_domains::__herbceptions_detail::
                __malloc_or_heapalloc_temp_buffer buffer;
            ::std::io_scatter_t scatters[4];
            ::std::size_t scatterlen{};
            if (is_itanium_cxx_eh) // is a C++ exception from the g++/clang++
                                   // ABI
            {
              // Raw mangled form, mirroring the MSVC sibling's reporting of
              // typeinfo->mangled; no demangling, no heap allocation.
              char const *ehname{};
              ::std::size_t ehname_len{};
              if (::std::error_query_information::name != query &&
                  hdr->exceptionType) {
                ehname = hdr->exceptionType->name();
                ehname_len = ::std::strlen(ehname);
              }

              // what() lives on the std::exception subobject, not on
              // type_info. When the RTTI graph proves a single-inheritance
              // path to std::exception, that subobject sits at offset 0 of
              // the thrown object, so dispatch directly through it.
              char const *ehmessage{};
              ::std::size_t ehmessage_len{};
              if (thrown && ::std::error_query_information::name != query &&
                  rtti_si_derives_from(
                      hdr->exceptionType,
                      __builtin_addressof(typeid(::std::exception)))) {
                auto *se{static_cast<::std::exception *>(thrown)};
                ehmessage = se->what();
                if (ehmessage) {
                  ehmessage_len = ::std::strlen(ehmessage);
                }
              }

              auto [name, message] =
                  itanium_exception_writestr(ehname, ehname_len, ehmessage,
                                             ehmessage_len, encoding, buffer);
              switch (static_cast<::std::uint_least32_t>(query)) {
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name): {
                *scatters = itanium_exception_name(encoding);
                scatterlen = 1u;
                break;
              }
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::message): {
                if (name.len) {
                  *scatters = left_parenthese(encoding);
                  scatters[1] = name;
                  scatters[2] = right_parenthese(encoding);
                  scatterlen = 3u;
                } else {
                  *scatters =
                      unknown_itanium_exception_message_partial(encoding);
                  scatterlen = 1u;
                }
                if (message.len) {
                  scatters[scatterlen] = message;
                  ++scatterlen;
                }
                break;
              }
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message): {
                if (name.len) {
                  *scatters =
                      known_itanium_exception_name_message_partial(encoding);
                  scatters[1] = name;
                  scatters[2] = right_parenthese(encoding);
                  scatterlen = 3u;
                } else {
                  *scatters = unknown_itanium_exception_name_message(encoding);
                  scatterlen = 1u;
                }
                if (message.len) {
                  scatters[scatterlen] = message;
                  ++scatterlen;
                }
                break;
              }
              default:
                return;
              }
            } else {
              switch (static_cast<::std::uint_least32_t>(query)) {
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name):
                *scatters = itanium_exception_name(encoding);
                break;
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::message):
                *scatters = unknown_itanium_exception_message_partial(encoding);
                break;
              case static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message):
                *scatters = unknown_itanium_exception_name_message(encoding);
                break;
              default:
                return;
              }
              scatterlen = 1u;
            }
            cookfun(cookie, scatters, scatterlen);
          }
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      void *thrown{reinterpret_cast<void *>(cd)};
      if (!thrown) {
        return ::std::errc::io_error;
      }
      auto *hdr{itanium_cxa_exception_from_thrown_object(thrown)};
      // system_error carries its own mapping: when the code belongs to the
      // generic category, the value IS the errc (errno) number.
      {
        void *se_subobj{};
        if (itanium_cxa_try_catch(hdr->exceptionType, thrown,
                                  typeid(::std::system_error), &se_subobj)) {
          auto *se{static_cast<::std::system_error *>(se_subobj)};
          ::std::error_code const &ec{se->code()};
          if (ec.category() == ::std::generic_category()) {
            return static_cast<::std::errc>(ec.value());
          }
          // Custom-category system_error: no errc meaning.
          return ::std::errc::io_error;
        }
      }
      // Catch-ladder semantics: the first kind the thrown object could be
      // caught as decides the errc mapping.
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::invalid_argument))) {
        return ::std::errc::invalid_argument;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::domain_error))) {
        return ::std::errc::argument_out_of_domain;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::length_error))) {
        return ::std::errc::value_too_large;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::out_of_range))) {
        return ::std::errc::result_out_of_range;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::overflow_error))) {
        return ::std::errc::value_too_large;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::underflow_error))) {
        return ::std::errc::value_too_large;
      }
      if (itanium_cxa_catchable(hdr->exceptionType, thrown,
                                typeid(::std::bad_alloc))) {
        return ::std::errc::not_enough_memory;
      }
      // runtime_error, system_error, std::exception itself and unknown
      // user types carry no errc meaning here.
      return ::std::errc::io_error;
    }
#if defined(__cpp_exceptions) && !defined(_MSC_VER)
    ,
    .do_throw_dynamic_exception =
        [](::std::size_t __cd, ::std::dynamic_exception_abi __ehabi) {
          if (__ehabi != ::std::dynamic_exception_abi::platform)
            return;
          void *__obj{reinterpret_cast<void *>(__cd)};
          if (!__obj) {
            return;
          }
#ifdef _LIBCPPABI_VERSION
          ::__cxxabiv1::__cxa_rethrow_primary_exception(__obj);
#else
          // Raise the primary directly, deliberately NOT through
          // std::rethrow_exception: the eh_ptr machinery would wrap it in a
          // dependent exception whose cleanup releases references behind
          // this domain's back. The minted reference is the flight
          // reference here -- the catching runtime consumes it inside
          // __cxa_end_catch. A failed unwind falls through, letting
          // std::error::throw_dynamic_exception raise its system_error
          // fallback instead.
          auto *hdr{itanium_cxa_exception_from_thrown_object(__obj)};
          auto *globals{reinterpret_cast<itanium_cxa_eh_globals *>(
              ::__cxxabiv1::__cxa_get_globals())};
          ++globals->uncaughtExceptions;
          _Unwind_RaiseException(&hdr->unwindHeader);
          --globals->uncaughtExceptions;
#endif
        }
#endif
};
} // namespace

} // namespace std::error_domains

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_itanium_exception_ptr() noexcept {
  return __builtin_addressof(
      ::std::error_domains::itanium_exception_ptr_domain);
}

#endif
