//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_syncbuf;

// basic_syncbuf& operator=(basic_syncbuf&& rhs);

#include <syncstream>
#include <sstream>
#include <cassert>
#include <concepts>

#include "test_macros.h"

template <class T, class propagate>
struct test_allocator : std::allocator<T> {
  using propagate_on_container_move_assignment = propagate;

  int id{-1};

  test_allocator(int _id = -1) : id(_id) {}
  test_allocator(test_allocator const& other)            = default;
  test_allocator(test_allocator&& other)                 = default;
  test_allocator& operator=(const test_allocator& other) = default;

  test_allocator& operator=(test_allocator&& other) {
    if constexpr (propagate_on_container_move_assignment::value)
      id = other.id;
    else
      id = -1;
    return *this;
  }
};

template <class T>
class test_buf : public std::basic_streambuf<T> {
public:
  int id;

  test_buf(int _id = 0) : id(_id) {}

  T* _pptr() { return this->pptr(); }
};

template <class T, class Alloc = std::allocator<T>>
class test_syncbuf : public std::basic_syncbuf<T, std::char_traits<T>, Alloc> {
  using Base = std::basic_syncbuf<T, std::char_traits<T>, Alloc>;

public:
  test_syncbuf() = default;

  test_syncbuf(test_buf<T>* buf, Alloc alloc) : Base(buf, alloc) {}

  test_syncbuf(typename Base::streambuf_type* buf, Alloc alloc) : Base(buf, alloc) {}

  void _setp(T* begin, T* end) { return this->setp(begin, end); }
};

// Helper wrapper to inspect the internal state of the basic_syncbuf
//
// This is used to validate some standard requirements and libc++
// implementation details.
template <class CharT, class Traits, class Allocator>
class syncbuf_inspector : public std::basic_syncbuf<CharT, Traits, Allocator> {
public:
  syncbuf_inspector() = default;
  explicit syncbuf_inspector(std::basic_syncbuf<CharT, Traits, Allocator>&& base)
      : std::basic_syncbuf<CharT, Traits, Allocator>(std::move(base)) {}

  void operator=(std::basic_syncbuf<CharT, Traits, Allocator>&& base) { *this = std::move(base); }

  using std::basic_syncbuf<CharT, Traits, Allocator>::pbase;
  using std::basic_syncbuf<CharT, Traits, Allocator>::pptr;
  using std::basic_syncbuf<CharT, Traits, Allocator>::epptr;
};

template <class CharT>
static void test_assign() {
  test_buf<CharT> base;

  { // Test using the real class, propagating allocator.
    using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, test_allocator<CharT, std::true_type>>;

    BuffT buff1(&base, test_allocator<CharT, std::true_type>{42});
    buff1.sputc(CharT('A'));

    assert(buff1.get_wrapped() != nullptr);

    BuffT buff2;
    assert(buff2.get_allocator().id == -1);
    buff2 = std::move(buff1);
    assert(buff1.get_wrapped() == nullptr);
    assert(buff2.get_wrapped() == &base);

    assert(buff2.get_wrapped() == &base);
    assert(buff2.get_allocator().id == 42);
  }

  { // Test using the real class, non-propagating allocator.
    using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, test_allocator<CharT, std::false_type>>;

    BuffT buff1(&base, test_allocator<CharT, std::false_type>{42});
    buff1.sputc(CharT('A'));

    assert(buff1.get_wrapped() != nullptr);

    BuffT buff2;
    assert(buff2.get_allocator().id == -1);
    buff2 = std::move(buff1);
    assert(buff1.get_wrapped() == nullptr);
    assert(buff2.get_wrapped() == &base);

    assert(buff2.get_wrapped() == &base);
    assert(buff2.get_allocator().id == -1);
  }

  { // Move assignment propagating allocator
    // Test using the inspection wrapper.
    // Not all these requirements are explicitly in the Standard,
    // however the asserts are based on secondary requirements. The
    // LIBCPP_ASSERTs are implementation specific.

    using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

    using Inspector = syncbuf_inspector<CharT, std::char_traits<CharT>, std::allocator<CharT>>;
    Inspector inspector1{BuffT(&base)};
    inspector1.sputc(CharT('A'));

    assert(inspector1.get_wrapped() != nullptr);
    assert(inspector1.pbase() != nullptr);
    assert(inspector1.pptr() != nullptr);
    assert(inspector1.epptr() != nullptr);
    assert(inspector1.pbase() != inspector1.pptr());
    assert(inspector1.pptr() - inspector1.pbase() == 1);
    [[maybe_unused]] std::streamsize size = inspector1.epptr() - inspector1.pbase();

    Inspector inspector2;
    inspector2 = std::move(inspector1);

    assert(inspector1.get_wrapped() == nullptr);
    LIBCPP_ASSERT(inspector1.pbase() == nullptr);
    LIBCPP_ASSERT(inspector1.pptr() == nullptr);
    LIBCPP_ASSERT(inspector1.epptr() == nullptr);
    assert(inspector1.pbase() == inspector1.pptr());

    assert(inspector2.get_wrapped() == &base);
    LIBCPP_ASSERT(inspector2.pbase() != nullptr);
    LIBCPP_ASSERT(inspector2.pptr() != nullptr);
    LIBCPP_ASSERT(inspector2.epptr() != nullptr);
    assert(inspector2.pptr() - inspector2.pbase() == 1);
    LIBCPP_ASSERT(inspector2.epptr() - inspector2.pbase() == size);
  }
}

template <class CharT>
static void test_basic() {
  { // Test properties
    std::basic_syncbuf<CharT> sync_buf1(nullptr);
    std::basic_syncbuf<CharT> sync_buf2(nullptr);
    [[maybe_unused]] std::same_as<std::basic_syncbuf<CharT>&> decltype(auto) ret =
        sync_buf1.operator=(std::move(sync_buf2));
  }

  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr1) == 1);
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr2) == 1);
#endif

    sync_buf2 = std::move(sync_buf1);
    assert(sync_buf2.get_wrapped() == &sstr1);

    assert(sstr1.str().empty());
    assert(sstr2.str() == expected);

#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr1) == 1);
    assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&sstr2) == 0);
#endif
  }

  assert(sstr1.str().size() == 1);
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr2.str() == expected);
}

template <class CharT>
static void test_short_write_after_assign() {
  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

    sync_buf2 = std::move(sync_buf1);
    sync_buf2.sputc(CharT('Z'));

    assert(sstr1.str().empty());
    assert(sstr2.str() == expected);
  }

  assert(sstr1.str().size() == 2);
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr1.str()[1] == CharT('Z'));
  assert(sstr2.str() == expected);
}

template <class CharT>
static void test_long_write_after_assign() {
  std::basic_stringbuf<CharT> sstr1;
  std::basic_stringbuf<CharT> sstr2;
  std::basic_string<CharT> expected(42, CharT('*')); // a long string

  {
    std::basic_syncbuf<CharT> sync_buf1(&sstr1);
    sync_buf1.sputc(CharT('A')); // a short string

    std::basic_syncbuf<CharT> sync_buf2(&sstr2);
    sync_buf2.sputn(expected.data(), expected.size());

    sync_buf2 = std::move(sync_buf1);
    sync_buf2.sputn(expected.data(), expected.size());

    assert(sstr1.str().empty());
    assert(sstr2.str() == expected);
  }

  assert(sstr1.str().size() == 1 + expected.size());
  assert(sstr1.str()[0] == CharT('A'));
  assert(sstr1.str().substr(1) == expected);
  assert(sstr2.str() == expected);
}

template <class CharT>
static void test_emit_on_assign() {
  { // don't emit / don't emit

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(false);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(false);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf2 = std::move(sync_buf1);
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // don't emit / do emit

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(true);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(false);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf2 = std::move(sync_buf1);
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().size() == 1);
      assert(sstr1.str()[0] == CharT('A'));
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // do emit / don't emit

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(false);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(true);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf2 = std::move(sync_buf1);
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }

  { // do emit / do emit

    std::basic_stringbuf<CharT> sstr1;
    std::basic_stringbuf<CharT> sstr2;
    std::basic_string<CharT> expected(42, CharT('*')); // a long string

    {
      std::basic_syncbuf<CharT> sync_buf1(&sstr1);
      sync_buf1.set_emit_on_sync(true);
      sync_buf1.sputc(CharT('A')); // a short string

      std::basic_syncbuf<CharT> sync_buf2(&sstr2);
      sync_buf2.set_emit_on_sync(true);
      sync_buf2.sputn(expected.data(), expected.size());

      sync_buf2 = std::move(sync_buf1);
      assert(sstr1.str().empty());
      assert(sstr2.str() == expected);

      sync_buf2.pubsync();
      assert(sstr1.str().size() == 1);
      assert(sstr1.str()[0] == CharT('A'));
      assert(sstr2.str() == expected);
    }

    assert(sstr1.str().size() == 1);
    assert(sstr1.str()[0] == CharT('A'));
    assert(sstr2.str() == expected);
  }
}

template <class CharT>
static void test() {
  test_assign<CharT>();
  test_basic<CharT>();
  test_short_write_after_assign<CharT>();
  test_long_write_after_assign<CharT>();
  test_emit_on_assign<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
