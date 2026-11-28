//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Verify that std::copy, std::move, std::copy_n, std::copy_backward, std::move_backward, and std::__unwrap_range
// validate __bounded_iter, __static_bounded_iter, and user-defined contiguous_iterator ranges up front (before writing
// any elements when lowering to memmove), and keep bounded output iterators wrapped when copying from non-sized input
// ranges (https://llvm.org/PR78771).

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: c++03
// Continuing past a failed check is undefined behavior here.
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

#include <__iterator/bounded_iter.h>
#include <__iterator/static_bounded_iter.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <list>
#include <type_traits>

#include "check_assertion.h"
#include "test_macros.h"

#if TEST_STD_VER >= 20
#  include <ranges>
#endif

#if __has_include(<sys/mman.h>)
#  include <sys/mman.h>
#  include <unistd.h>
#  ifndef MAP_ANONYMOUS
#    define MAP_ANONYMOUS MAP_ANON
#  endif
#  define TEST_HAS_MMAP 1
#else
#  define TEST_HAS_MMAP 0
#endif

namespace {

constexpr int kCanary = 0x5A5A5A5A;

// Allocates a MAP_SHARED page (when available) so any writes in a forked death-test child before trapping are visible
// in the parent after the child terminates.
struct SharedIntBuffer {
  std::size_t count_;
  std::size_t byte_size_ = 0;
  int* data_             = nullptr;
#if !TEST_HAS_MMAP
  int fallback_[16] = {};
#endif

  explicit SharedIntBuffer(std::size_t count = 8) : count_(count) {
#if TEST_HAS_MMAP
    const std::size_t page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    byte_size_                  = ((count * sizeof(int) + page_size - 1) / page_size) * page_size;
    data_ = static_cast<int*>(::mmap(nullptr, byte_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    assert(data_ != MAP_FAILED);
#else
    assert(count <= 16);
    data_ = fallback_;
#endif
    for (std::size_t i = 0; i < count_; ++i)
      data_[i] = kCanary;
  }

  SharedIntBuffer(const SharedIntBuffer&)            = delete;
  SharedIntBuffer& operator=(const SharedIntBuffer&) = delete;

  ~SharedIntBuffer() {
#if TEST_HAS_MMAP
    if (data_ != MAP_FAILED)
      ::munmap(data_, byte_size_);
#endif
  }

  void assert_untouched(std::size_t from = 0, std::size_t to = static_cast<std::size_t>(-1)) const {
    if (to == static_cast<std::size_t>(-1))
      to = count_;
    for (std::size_t i = from; i < to; ++i)
      assert(data_[i] == kCanary);
  }
};

#if TEST_STD_VER >= 20
// User-defined C++20 contiguous_iterator modeled on Chromium's CheckedContiguousIterator (PR78771).
template <class T>
class CustomCheckedIter {
  const T* start_ = nullptr;
  T* current_     = nullptr;
  const T* end_   = nullptr;

public:
  using difference_type   = std::ptrdiff_t;
  using value_type        = std::remove_cv_t<T>;
  using pointer           = T*;
  using reference         = T&;
  using iterator_category = std::contiguous_iterator_tag;
  using iterator_concept  = std::contiguous_iterator_tag;

  constexpr CustomCheckedIter() = default;
  constexpr CustomCheckedIter(const T* start, T* current, const T* end) : start_(start), current_(current), end_(end) {}

  constexpr reference operator*() const {
    if (current_ == end_)
      std::abort();
    return *current_;
  }
  constexpr pointer operator->() const {
    if (current_ == end_)
      std::abort();
    return current_;
  }
  constexpr reference operator[](difference_type n) const { return *(*this + n); }
  constexpr CustomCheckedIter& operator++() { return *this += 1; }
  constexpr CustomCheckedIter operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }
  constexpr CustomCheckedIter& operator--() { return *this -= 1; }
  constexpr CustomCheckedIter operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }
  constexpr CustomCheckedIter& operator+=(difference_type n) {
    if (n < start_ - current_ || n > end_ - current_)
      std::abort();
    current_ += n;
    return *this;
  }
  constexpr CustomCheckedIter& operator-=(difference_type n) {
    if (n < current_ - end_ || n > current_ - start_)
      std::abort();
    current_ -= n;
    return *this;
  }
  friend constexpr CustomCheckedIter operator+(CustomCheckedIter it, difference_type n) { return it += n; }
  friend constexpr CustomCheckedIter operator+(difference_type n, CustomCheckedIter it) { return it += n; }
  friend constexpr CustomCheckedIter operator-(CustomCheckedIter it, difference_type n) { return it -= n; }
  friend constexpr difference_type operator-(const CustomCheckedIter& lhs, const CustomCheckedIter& rhs) {
    if (lhs.start_ != rhs.start_ || lhs.end_ != rhs.end_)
      std::abort();
    return lhs.current_ - rhs.current_;
  }
  friend constexpr bool operator==(const CustomCheckedIter& lhs, const CustomCheckedIter& rhs) {
    if (lhs.start_ != rhs.start_ || lhs.end_ != rhs.end_)
      std::abort();
    return lhs.current_ == rhs.current_;
  }
  friend constexpr auto operator<=>(const CustomCheckedIter& lhs, const CustomCheckedIter& rhs) {
    if (lhs.start_ != rhs.start_ || lhs.end_ != rhs.end_)
      std::abort();
    return lhs.current_ <=> rhs.current_;
  }

  constexpr T* raw() const { return current_; }
};
#endif // TEST_STD_VER >= 20

} // namespace

#if TEST_STD_VER >= 20
template <class T>
struct std::pointer_traits<CustomCheckedIter<T>> {
  using pointer         = CustomCheckedIter<T>;
  using element_type    = T;
  using difference_type = std::ptrdiff_t;
  static constexpr element_type* to_address(pointer p) noexcept { return p.raw(); }
};
#endif

int main(int, char**) {
  const int src[6]     = {1, 2, 3, 4, 5, 6};
  const auto src_begin = std::__make_bounded_iter(src + 0, src + 0, src + 6);
  const auto src_end   = std::__make_bounded_iter(src + 6, src + 0, src + 6);

  // Forward copy / move / copy_n into undersized __bounded_iter traps before memmove writes any elements.
  {
    SharedIntBuffer shared;
    auto dst_begin = std::__make_bounded_iter(shared.data_, shared.data_, shared.data_ + 4);

    TEST_LIBCPP_ASSERT_FAILURE(std::copy(src_begin, src_end, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();

    TEST_LIBCPP_ASSERT_FAILURE(std::move(src_begin, src_end, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();

    TEST_LIBCPP_ASSERT_FAILURE(std::copy_n(src_begin, 5, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();

#if TEST_STD_VER >= 20
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::copy(src_begin, src_end, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::move(src_begin, src_end, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::copy_n(src_begin, 5, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();
#endif
  }

  // Backward copy / move into undersized __bounded_iter traps before memmove writes any elements.
  {
    SharedIntBuffer shared;
    auto dst_end = std::__make_bounded_iter(shared.data_ + 8, shared.data_ + 4, shared.data_ + 8);

    TEST_LIBCPP_ASSERT_FAILURE(std::copy_backward(src_begin, src_end, dst_end),
                               "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    shared.assert_untouched();

    TEST_LIBCPP_ASSERT_FAILURE(std::move_backward(src_begin, src_end, dst_end),
                               "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    shared.assert_untouched();

#if TEST_STD_VER >= 20
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::copy_backward(src_begin, src_end, dst_end),
                               "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::move_backward(src_begin, src_end, dst_end),
                               "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    shared.assert_untouched();
#endif
  }

  // Invalid source ranges (first > last or past source end), including other __unwrap_range callers.
  {
    SharedIntBuffer shared;
    auto dst_begin = std::__make_bounded_iter(shared.data_, shared.data_, shared.data_ + 8);
    auto short_src = std::__make_bounded_iter(src + 0, src + 0, src + 3);

    TEST_LIBCPP_ASSERT_FAILURE(
        std::copy(src_end, src_begin, dst_begin), "__unwrap_range: [first, last) must be a valid range");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::copy_backward(src_end, src_begin, dst_begin + 8), "__unwrap_range: [first, last) must be a valid range");
    TEST_LIBCPP_ASSERT_FAILURE(std::copy(short_src, src_end, dst_begin),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();

#if TEST_STD_VER >= 20
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::find(src_end, src_begin, 1), "__unwrap_range: [first, last) must be a valid range");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::find(short_src, src_end, 6),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::equal(short_src, src_end, src_begin, src_end),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::mismatch(short_src, src_end, src_begin, src_end),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::lexicographical_compare(short_src, src_end, src_begin, src_end),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
#endif
  }

  // Non-sized input range (std::list) keeps bounded output iterator wrapped and never writes out of bounds.
  {
    SharedIntBuffer shared_fwd, shared_bwd;
    const std::list<int> list_src(src, src + 6);
    auto dst_begin = std::__make_bounded_iter(shared_fwd.data_, shared_fwd.data_, shared_fwd.data_ + 4);
    auto dst_end   = std::__make_bounded_iter(shared_bwd.data_ + 8, shared_bwd.data_ + 4, shared_bwd.data_ + 8);

    TEST_LIBCPP_ASSERT_FAILURE(std::copy(list_src.begin(), list_src.end(), dst_begin),
                               "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
    shared_fwd.assert_untouched(4, 8);

    TEST_LIBCPP_ASSERT_FAILURE(std::copy_backward(list_src.begin(), list_src.end(), dst_end),
                               "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    shared_bwd.assert_untouched(0, 4);
  }

  // Doubly-reversed bounded output iterator and __static_bounded_iter.
  {
    using RevIt    = std::reverse_iterator<std::__bounded_iter<int*> >;
    using RevRevIt = std::reverse_iterator<RevIt>;

    SharedIntBuffer shared;
    RevRevIt rr_dst(RevIt(std::__make_bounded_iter(shared.data_, shared.data_, shared.data_ + 4)));
    RevRevIt rr_back(RevIt(std::__make_bounded_iter(shared.data_ + 8, shared.data_ + 4, shared.data_ + 8)));

    TEST_LIBCPP_ASSERT_FAILURE(std::copy(src_begin, src_end, rr_dst),
                               "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::copy_backward(src_begin, src_end, rr_back),
                               "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");

    auto s_src_begin = std::__make_static_bounded_iter<6>(src + 0, src + 0);
    auto s_src_end   = std::__make_static_bounded_iter<6>(src + 6, src + 0);
    auto s_dst_begin = std::__make_static_bounded_iter<4>(shared.data_, shared.data_);
    auto s_dst_end   = std::__make_static_bounded_iter<4>(shared.data_ + 8, shared.data_ + 4);

    TEST_LIBCPP_ASSERT_FAILURE(std::copy(s_src_begin, s_src_end, s_dst_begin),
                               "__static_bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::move_backward(s_src_begin, s_src_end, s_dst_end),
                               "__static_bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::copy(s_src_end, s_src_begin, s_dst_begin), "__unwrap_range: [first, last) must be a valid range");

    auto s_short_src = std::__make_static_bounded_iter<3>(src + 0, src + 0);
    auto s_other_end = std::__make_static_bounded_iter<3>(src + 6, src + 3);
    auto s_wide_dst  = std::__make_static_bounded_iter<6>(shared.data_, shared.data_);
    TEST_LIBCPP_ASSERT_FAILURE(std::copy(s_short_src, s_other_end, s_wide_dst),
                               "__static_bounded_iter::operator+=: Attempt to advance an iterator past the end");
    shared.assert_untouched();
  }

#if TEST_STD_VER >= 20 && (!defined(TEST_HAS_NO_INT128) || __SIZEOF_POINTER__ < __SIZEOF_LONG_LONG__)
  // Wider input difference_type whose static_cast<ptrdiff_t>(n) wraps to -1 traps before writing even at begin + 1.
  {
    SharedIntBuffer shared;
    auto dst_mid = std::__make_bounded_iter(shared.data_ + 1, shared.data_ + 0, shared.data_ + 4);
    auto huge    = std::views::iota(std::uintptr_t{0}, ~std::uintptr_t{0});

    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::copy(huge, dst_mid), "std::copy/move: [first, last) must be a valid range");
    shared.assert_untouched();
  }
#endif

#if TEST_STD_VER >= 20
  // User-defined C++20 contiguous_iterator (PR78771) also traps before memmove writes any elements.
  {
    SharedIntBuffer shared;
    CustomCheckedIter<int> cci_begin(shared.data_, shared.data_, shared.data_ + 4);
    CustomCheckedIter<int> cci_back_end(shared.data_ + 4, shared.data_ + 8, shared.data_ + 8);
    int other[4] = {1, 2, 3, 4};
    CustomCheckedIter<int> other_end(other, other + 4, other + 4);

    EXPECT_ANY_DEATH(std::copy(src, src + 6, cci_begin));
    EXPECT_ANY_DEATH(std::copy_backward(src, src + 6, cci_back_end));
    EXPECT_ANY_DEATH(std::copy(cci_begin, other_end, shared.data_));
    shared.assert_untouched();
  }
#endif

  return 0;
}
