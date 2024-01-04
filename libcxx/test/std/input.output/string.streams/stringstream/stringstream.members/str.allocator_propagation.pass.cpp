//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// This test ensures that we properly propagate allocators from stringstream's
// inner string object to the new string returned from .str().
// `str() const&` is specified to preserve the allocator (not copy the string).
// `str() &&` isn't specified, but should preserve the allocator (move the string).

#include <cassert>
#include <memory>
#include <memory_resource>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include "make_string.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class CharT>
void test_soccc_behavior() {
  using Alloc = SocccAllocator<CharT>;
  using SS    = std::basic_stringstream<CharT, std::char_traits<CharT>, Alloc>;
  using S     = std::basic_string<CharT, std::char_traits<CharT>, Alloc>;
  {
    SS ss = SS(std::ios_base::out, Alloc(10));

    // [stringbuf.members]/6 specifies that the allocator is copied,
    // not select_on_container_copy_construction'ed.
    //
    S copied = ss.str();
    assert(copied.get_allocator().count_ == 10);
    assert(ss.rdbuf()->get_allocator().count_ == 10);
    assert(copied.empty());

    // sanity-check that SOCCC does in fact work
    assert(S(copied).get_allocator().count_ == 11);

    // [stringbuf.members]/10 doesn't specify the allocator to use,
    // but copying the allocator as-if-by moving the string makes sense.
    //
    S moved = std::move(ss).str();
    assert(moved.get_allocator().count_ == 10);
    assert(ss.rdbuf()->get_allocator().count_ == 10);
    assert(moved.empty());
  }
}

template <class CharT,
          class Base = std::basic_stringbuf<CharT, std::char_traits<CharT>, std::pmr::polymorphic_allocator<CharT>>>
struct StringBuf : Base {
  explicit StringBuf(std::pmr::memory_resource* mr) : Base(std::ios_base::in, mr) {}
  void public_setg(int a, int b, int c) {
    CharT* p = this->eback();
    assert(this->view().data() == p);
    this->setg(p + a, p + b, p + c);
    assert(this->eback() == p + a);
    assert(this->view().data() == p + a);
  }
};

template <class CharT>
void test_allocation_is_pilfered() {
  using SS = std::basic_stringstream<CharT, std::char_traits<CharT>, std::pmr::polymorphic_allocator<CharT>>;
  using S  = std::pmr::basic_string<CharT>;
  alignas(void*) char buf[80 * sizeof(CharT)];
  const CharT* initial =
      MAKE_CSTRING(CharT, "a very long string that exceeds the small string optimization buffer length");
  {
    std::pmr::set_default_resource(std::pmr::null_memory_resource());
    auto mr1 = std::pmr::monotonic_buffer_resource(buf, sizeof(buf), std::pmr::null_memory_resource());
    SS ss    = SS(S(initial, &mr1));
    S s      = std::move(ss).str();
    assert(s == initial);
  }
  {
    // Try moving-out-of a stringbuf whose view() is not the entire string.
    // This is libc++'s behavior; libstdc++ doesn't allow such stringbufs to be created.
    //
    std::pmr::set_default_resource(std::pmr::null_memory_resource());
    auto mr1 = std::pmr::monotonic_buffer_resource(buf, sizeof(buf), std::pmr::null_memory_resource());
    auto src = StringBuf<CharT>(&mr1);
    src.str(S(initial, &mr1));
    src.public_setg(2, 6, 40);
    SS ss(std::ios_base::in, &mr1);
    *ss.rdbuf() = std::move(src);
    LIBCPP_ASSERT(ss.view() == std::basic_string_view<CharT>(initial).substr(2, 38));
    S s = std::move(ss).str();
    LIBCPP_ASSERT(s == std::basic_string_view<CharT>(initial).substr(2, 38));
  }
}

template <class CharT>
void test_no_foreign_allocations() {
  using SS = std::basic_stringstream<CharT, std::char_traits<CharT>, std::pmr::polymorphic_allocator<CharT>>;
  using S  = std::pmr::basic_string<CharT>;
  const CharT* initial =
      MAKE_CSTRING(CharT, "a very long string that exceeds the small string optimization buffer length");
  {
    std::pmr::set_default_resource(std::pmr::null_memory_resource());
    auto mr1 = std::pmr::monotonic_buffer_resource(std::pmr::new_delete_resource());
    auto ss  = SS(S(initial, &mr1));
    assert(ss.rdbuf()->get_allocator().resource() == &mr1);

    // [stringbuf.members]/6 specifies that the result of `str() const &`
    // does NOT use the default allocator; it uses the original allocator.
    //
    S copied = ss.str();
    assert(copied.get_allocator().resource() == &mr1);
    assert(ss.rdbuf()->get_allocator().resource() == &mr1);
    assert(copied == initial);

    // [stringbuf.members]/10 doesn't specify the allocator to use,
    // but copying the allocator as-if-by moving the string makes sense.
    //
    S moved = std::move(ss).str();
    assert(moved.get_allocator().resource() == &mr1);
    assert(ss.rdbuf()->get_allocator().resource() == &mr1);
    assert(moved == initial);
  }
}

int main(int, char**) {
  test_soccc_behavior<char>();
  test_allocation_is_pilfered<char>();
  test_no_foreign_allocations<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_soccc_behavior<wchar_t>();
  test_allocation_is_pilfered<wchar_t>();
  test_no_foreign_allocations<wchar_t>();
#endif

  return 0;
}
