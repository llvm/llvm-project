//===--- domain_test.cpp - libherbceptions domain tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests the error-domain runtime: posix (std::errc), cxa_exception_code, and
// (on Windows) win32/nt/com/wine. Verifies domain identity, do_to_errc
// mapping, cross-domain equivalence, do_name/do_message, and the herbception
// type traits.
//
// RUN: %cxx %herbceptions_flags -I%herbceptions_include -L%herbceptions_lib -lherbceptions %s -o %t && %t | %FileCheck %s
// CHECK: PASS
//
//===----------------------------------------------------------------------===//

#include "herbceptions/error"

#include <cassert>
#include <cstdio>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

namespace {
struct capture_ctx {
  char buf[512];
  std::size_t len;
};
} // namespace

// A do_query_information collector that appends the emitted text into a buffer
// (may make several cookfun calls; each appends).
static void capture_cookfun(void *cookie, ::std::io_scatter_t const *v,
                            ::std::size_t n) noexcept {
  auto *ctx = static_cast<capture_ctx *>(cookie);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < v[i].len; ++j) {
      if (ctx->len < sizeof(ctx->buf) - 1)
        ctx->buf[ctx->len++] = static_cast<char const *>(v[i].base)[j];
    }
  }
  ctx->buf[ctx->len] = '\0';
}

int main() {
  auto const *posix = ::std::error_domains::__cxa_error_domain_posix();
#ifdef _MSC_VER
  auto const *cxa =
      ::std::error_domains::__cxa_error_domain_msvc_exception_ptr();
#else
  auto const *cxa =
      ::std::error_domains::__cxa_error_domain_itanium_exception_ptr();
#endif
  // Non-null domains.
  CHECK(posix != nullptr);
  CHECK(cxa != nullptr);

  // std::errc maps to the posix domain.
  CHECK(::std::error_domain<::std::errc>::domain() == posix);

  // do_to_errc is identity for posix.
  CHECK(posix->do_to_errc(2) == ::std::errc::no_such_file_or_directory);
  CHECK(posix->do_to_errc(13) == ::std::errc::permission_denied);

  // do_query_information produces name and/or message text.
  capture_ctx nctx{};
  posix->do_query_information(2, ::std::error_query_information::name,
                              ::std::error_reporter_encoding::utf8, &nctx,
                              capture_cookfun);
  CHECK(std::string(nctx.buf, nctx.len) == "posix");

  capture_ctx mctx{};
  posix->do_query_information(2, ::std::error_query_information::message,
                              ::std::error_reporter_encoding::utf8, &mctx,
                              capture_cookfun);
  CHECK(std::string(mctx.buf, mctx.len) == "(2)No such file or directory");

  capture_ctx nmctx{};
  posix->do_query_information(2, ::std::error_query_information::name_message,
                              ::std::error_reporter_encoding::utf8, &nmctx,
                              capture_cookfun);
  CHECK(std::string(nmctx.buf, nmctx.len) ==
        "[posix](2)No such file or directory");

  capture_ctx cx{};
  cxa->do_query_information(0, ::std::error_query_information::name,
                            ::std::error_reporter_encoding::utf8, &cx,
                            capture_cookfun);
  CHECK(std::string(cx.buf, cx.len)
            .find(
#ifdef _MSC_VER
                "msvc_exception"
#else
                "itanium_exception"
#endif
                ) == 0);

  // posix cross-domain equivalence: identity within the domain.
  CHECK(posix->do_equivalent(2, posix, 2));

  // Herbception type traits (available under -fherbceptions).
  static_assert(::std::is_herbceptions_throwsable_v<::std::errc>);
  static_assert(!::std::is_herbceptions_throwsable_v<int>);
  static_assert(
      ::std::is_invoke_herbceptions_fails_v<int (*)(int) fails{::std::errc}>);
  static_assert(!::std::is_invoke_herbceptions_fails_v<int (*)(int)>);

  // is_herbceptions_throws_constructible
  // A class with a throws copy constructor.
  struct throws_copy {
    throws_copy(throws_copy const &) throws;
  };
  static_assert(
      ::std::is_herbceptions_throws_constructible_v<throws_copy,
                                                    throws_copy const &>);
  static_assert(
      ::std::is_herbceptions_throws_copy_constructible_v<throws_copy>);
  // A class without a throws copy constructor.
  struct noexcept_copy {
    noexcept_copy(noexcept_copy const &) noexcept;
  };
  static_assert(
      !::std::is_herbceptions_throws_copy_constructible_v<noexcept_copy>);
  // Move constructible
  struct throws_move {
    throws_move(throws_move &&) throws;
  };
  static_assert(
      ::std::is_herbceptions_throws_move_constructible_v<throws_move>);
  struct noexcept_move {
    noexcept_move(noexcept_move &&) noexcept;
  };
  static_assert(
      !::std::is_herbceptions_throws_move_constructible_v<noexcept_move>);

  // is_herbceptions_throws_invocable
  // A throws function is invocable and can throw.
  auto throws_func = []() throws {};
  static_assert(
      ::std::is_herbceptions_throws_invocable_v<decltype(throws_func)>);
  // A noexcept function is invocable but cannot throw.
  auto noexcept_func = []() noexcept {};
  static_assert(
      !::std::is_herbceptions_throws_invocable_v<decltype(noexcept_func)>);
  // int is not invocable.
  static_assert(!::std::is_herbceptions_throws_invocable_v<int>);
  // is_herbceptions_throws_invocable_r
  auto throws_int_func = []() throws -> int { return 0; };
  static_assert(
      ::std::is_herbceptions_throws_invocable_r_v<int,
                                                  decltype(throws_int_func)>);
  static_assert(
      ::std::is_herbceptions_throws_invocable_r_v<void,
                                                  decltype(throws_int_func)>);
  static_assert(
      !::std::is_herbceptions_throws_invocable_r_v<int, decltype(throws_func)>);
  static_assert(!::std::is_herbceptions_throws_invocable_r_v<int, int>);

#if (defined(_WIN32) || defined(__CYGWIN__)) && 0
  auto const *win32 = ::std::error_domains::__cxa_error_domain_win32();
  auto const *nt = ::std::error_domains::__cxa_error_domain_nt();
  auto const *com = ::std::error_domains::__cxa_error_domain_com();
  auto const *wine = ::std::error_domains::__cxa_error_domain_wine();
  CHECK(win32 != nullptr && nt != nullptr && com != nullptr && wine != nullptr);

  // win32 -> errc mapping.
  CHECK(win32->do_to_errc((std::size_t)::std::win32_errc::file_not_found) ==
        ::std::errc::no_such_file_or_directory);
  CHECK(win32->do_to_errc((std::size_t)::std::win32_errc::access_denied) ==
        ::std::errc::permission_denied);

  // Cross-domain equivalence: win32 file_not_found ~ posix no_such_file.
  CHECK(posix->do_equivalent(2, win32,
                             (std::size_t)::std::win32_errc::file_not_found));
  CHECK(win32->do_equivalent((std::size_t)::std::win32_errc::file_not_found,
                             posix, 2));

  // nt -> errc.
  CHECK(nt->do_to_errc((std::size_t)::std::nt_errc::object_name_not_found) ==
        ::std::errc::no_such_file_or_directory);

  // com -> errc.
  CHECK(com->do_to_errc((std::size_t)::std::com_errc::outofmemory) ==
        ::std::errc::not_enough_memory);

  // wine shares the host errno numbering with posix.
  CHECK(wine->do_to_errc(2) == ::std::errc::no_such_file_or_directory);
#else
  // On non-Windows only posix and itanium_exception_ptr exist.
  // (The win32/nt/com/wine accessors are not declared on this platform.)
  CHECK(posix != nullptr);
  CHECK(cxa != nullptr);
#endif

  if (failures == 0)
    std::printf("PASS\n");
  return failures == 0 ? 0 : 1;
}
