//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <random>

// template<class R, class G>
//   requires output_range<R, invoke_result_t<G&>> && uniform_random_bit_generator<remove_cvref_t<G>>
//   constexpr borrowed_iterator_t<R> ranges::generate_random(R&& r, G&& g);
//
// template<class G, output_iterator<invoke_result_t<G&>> O, sentinel_for<O> S>
//   requires uniform_random_bit_generator<remove_cvref_t<G>>
//   constexpr O ranges::generate_random(O first, S last, G&& g);

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <deque>
#include <list>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

// A uniform random bit generator without a generate_random member. It produces 1, 2, 3, ...
struct CountingEngine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }

  constexpr result_type operator()() {
    ++calls;
    return next++;
  }

  result_type next = 1;
  int calls        = 0;
};
static_assert(std::uniform_random_bit_generator<CountingEngine>);

// Has a generate_random member that accepts any range ([alg.rand.generate]/1.1). The member produces the same
// values as invocations of operator() would ([alg.rand.generate]/3).
struct RangeMemberEngine : CountingEngine {
  template <class R>
  constexpr void generate_random(R&& r) {
    ++member_calls;
    for (auto&& x : r)
      x = next++;
  }

  int member_calls = 0;
};
static_assert(std::uniform_random_bit_generator<RangeMemberEngine>);

// Has a generate_random member that only accepts spans of result_type. Template argument deduction fails for other
// ranges, so g.generate_random(r) is ill-formed and this member can only be used through spans
// ([alg.rand.generate]/1.2).
struct SpanMemberEngine : CountingEngine {
  template <std::size_t Extent>
  constexpr void generate_random(std::span<result_type, Extent> s) {
    ++member_calls;
    elements_generated += static_cast<int>(s.size());
    for (auto& x : s)
      x = next++;
  }

  int member_calls       = 0;
  int elements_generated = 0;
};
static_assert(std::uniform_random_bit_generator<SpanMemberEngine>);

// operator() is const, so the generator can be passed as a const lvalue.
struct ConstEngine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }

  constexpr result_type operator()() const { return (*next)++; }

  result_type* next;
};
static_assert(std::uniform_random_bit_generator<const ConstEngine>);

template <class Range>
constexpr bool is_iota_from_one(const Range& r) {
  unsigned expected = 1;
  for (auto&& x : r) {
    if (x != expected)
      return false;
    ++expected;
  }
  return true;
}

// The engine has no generate_random member, so the algorithm must call g() once per element.
template <class Iter, class Sent>
constexpr void test_no_member() {
  { // (iterator, sentinel) overload
    std::array<unsigned, 5> a{};
    CountingEngine g;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.calls == 5);
  }
  { // (range) overload
    std::array<unsigned, 5> a{};
    CountingEngine g;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(std::move(range), g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.calls == 5);
  }
  { // empty range
    std::array<unsigned, 0> a{};
    CountingEngine g;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data())), g);
    assert(base(result) == a.data());
    assert(g.calls == 0);
  }
}

template <class Iter>
constexpr void test_no_member_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>)
    test_no_member<Iter, Iter>();
  test_no_member<Iter, sentinel_wrapper<Iter>>();
  test_no_member<Iter, sized_sentinel<Iter>>();
}

// The engine has a generate_random member that accepts the range, so it must be called exactly once, even for an
// empty range ([alg.rand.generate]/1.1).
//
// Move-only (single-pass) iterators are not tested here: when the range is not common, the result is
// ranges::next(ranges::begin(r), ranges::end(r)) ([algorithms.requirements]/13), which calls begin() again after
// the member function has consumed the range.
template <class Iter, class Sent>
constexpr void test_range_member() {
  { // (iterator, sentinel) overload
    std::array<unsigned, 5> a{};
    RangeMemberEngine g;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
  { // (range) overload
    std::array<unsigned, 5> a{};
    RangeMemberEngine g;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(range, g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
  { // empty range
    std::array<unsigned, 0> a{};
    RangeMemberEngine g;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data())), g);
    assert(base(result) == a.data());
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
}

template <class Iter>
constexpr void test_range_member_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>)
    test_range_member<Iter, Iter>();
  test_range_member<Iter, sentinel_wrapper<Iter>>();
  test_range_member<Iter, sized_sentinel<Iter>>();
}

// The engine only accepts spans. For sized ranges the algorithm may use the member through spans
// ([alg.rand.generate]/1.2); libc++ does so and never calls g(). For ranges that are not sized, the algorithm must
// call g() once per element and must not call the member ([alg.rand.generate]/1.3).
template <class Iter, class Sent>
constexpr void test_span_member() {
  constexpr bool sized = std::sized_sentinel_for<Sent, Iter>;
  { // (iterator, sentinel) overload
    std::array<unsigned, 5> a{};
    SpanMemberEngine g;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.calls + g.elements_generated == 5);
    if constexpr (sized) {
      LIBCPP_ASSERT(g.calls == 0);
    } else {
      assert(g.member_calls == 0);
    }
  }
  { // (range) overload
    std::array<unsigned, 5> a{};
    SpanMemberEngine g;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(std::move(range), g);
    assert(base(result) == a.data() + a.size());
    assert(is_iota_from_one(a));
    assert(g.calls + g.elements_generated == 5);
    if constexpr (sized) {
      LIBCPP_ASSERT(g.calls == 0);
    } else {
      assert(g.member_calls == 0);
    }
  }
  { // empty range
    std::array<unsigned, 0> a{};
    SpanMemberEngine g;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data())), g);
    assert(base(result) == a.data());
    assert(g.calls == 0);
    assert(g.elements_generated == 0);
  }
}

template <class Iter>
constexpr void test_span_member_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>)
    test_span_member<Iter, Iter>();
  test_span_member<Iter, sentinel_wrapper<Iter>>();
  test_span_member<Iter, sized_sentinel<Iter>>();
}

constexpr void test_iterators() {
  test_no_member_iter<cpp17_output_iterator<unsigned*>>();
  test_no_member_iter<cpp20_output_iterator<unsigned*>>();
  test_no_member_iter<cpp17_input_iterator<unsigned*>>();
  test_no_member_iter<forward_iterator<unsigned*>>();
  test_no_member_iter<bidirectional_iterator<unsigned*>>();
  test_no_member_iter<random_access_iterator<unsigned*>>();
  test_no_member_iter<contiguous_iterator<unsigned*>>();
  test_no_member_iter<unsigned*>();

  test_range_member_iter<cpp17_output_iterator<unsigned*>>();
  test_range_member_iter<cpp17_input_iterator<unsigned*>>();
  test_range_member_iter<forward_iterator<unsigned*>>();
  test_range_member_iter<bidirectional_iterator<unsigned*>>();
  test_range_member_iter<random_access_iterator<unsigned*>>();
  test_range_member_iter<contiguous_iterator<unsigned*>>();
  test_range_member_iter<unsigned*>();

  test_span_member_iter<cpp17_output_iterator<unsigned*>>();
  test_span_member_iter<cpp20_output_iterator<unsigned*>>();
  test_span_member_iter<cpp17_input_iterator<unsigned*>>();
  test_span_member_iter<forward_iterator<unsigned*>>();
  test_span_member_iter<bidirectional_iterator<unsigned*>>();
  test_span_member_iter<random_access_iterator<unsigned*>>();
  test_span_member_iter<contiguous_iterator<unsigned*>>();
  test_span_member_iter<unsigned*>();
}

constexpr void test_ranges() {
  { // Containers passed as lvalues return an iterator.
    std::vector<unsigned> v(10);
    CountingEngine g;
    std::same_as<std::vector<unsigned>::iterator> decltype(auto) result = std::ranges::generate_random(v, g);
    assert(result == v.end());
    assert(is_iota_from_one(v));
    assert(g.calls == 10);
  }
  { // Non-borrowed ranges passed as rvalues return ranges::dangling.
    CountingEngine g;
    std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::generate_random(std::array<unsigned, 3>{}, g);
    (void)result;
    assert(g.calls == 3);
  }
  { // The member function is called with the forwarded range.
    RangeMemberEngine g;
    std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::generate_random(std::array<unsigned, 3>{}, g);
    (void)result;
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
  { // A span is passed to the member function as is ([alg.rand.generate]/1.1).
    std::array<unsigned, 4> a{};
    SpanMemberEngine g;
    std::span<unsigned> s(a);
    std::same_as<std::span<unsigned>::iterator> decltype(auto) result = std::ranges::generate_random(s, g);
    assert(result == s.end());
    assert(is_iota_from_one(a));
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
  { // Contiguous range of result_type: the member is called with a span over the whole range.
    std::array<unsigned, 300> a{};
    SpanMemberEngine g;
    std::same_as<std::array<unsigned, 300>::iterator> decltype(auto) result = std::ranges::generate_random(a, g);
    assert(result == a.end());
    assert(is_iota_from_one(a));
    assert(g.calls + g.elements_generated == 300);
    LIBCPP_ASSERT(g.member_calls == 1);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // Different value type: the values are converted. Large enough to need more than one chunk.
    std::vector<unsigned long long> v(1000);
    SpanMemberEngine g;
    std::same_as<std::vector<unsigned long long>::iterator> decltype(auto) result = std::ranges::generate_random(v, g);
    assert(result == v.end());
    assert(is_iota_from_one(v));
    assert(g.calls + g.elements_generated == 1000);
    LIBCPP_ASSERT(g.member_calls >= 1);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // Non-common range with a sized sentinel: the result is the end iterator, not the sentinel.
    std::array<unsigned, 6> a{};
    RangeMemberEngine g;
    auto v = std::views::counted(forward_iterator<unsigned*>(a.data()), 4);
    static_assert(!std::ranges::common_range<decltype(v)>);
    std::same_as<std::counted_iterator<forward_iterator<unsigned*>>> decltype(auto) result =
        std::ranges::generate_random(v, g);
    assert(result.count() == 0);
    assert(base(result.base()) == a.data() + 4);
    assert((a == std::array<unsigned, 6>{1, 2, 3, 4, 0, 0}));
  }
  { // Proxy references: the chunked path writes through vector<bool>::reference.
    std::vector<bool> v(300);
    SpanMemberEngine g;
    g.next = 0;
    std::ranges::generate_random(v, g);
    assert(!v[0]);
    for (std::size_t i = 1; i != v.size(); ++i)
      assert(v[i]);
    assert(g.calls + g.elements_generated == 300);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // Values are converted to the value type of the range.
    std::array<double, 3> a{};
    CountingEngine g;
    std::ranges::generate_random(a, g);
    assert((a == std::array<double, 3>{1.0, 2.0, 3.0}));
  }
  { // The engine can be an rvalue.
    std::array<unsigned, 3> a{};
    std::ranges::generate_random(a, CountingEngine{});
    assert(is_iota_from_one(a));
    std::ranges::generate_random(a.begin(), a.end(), SpanMemberEngine{});
    assert(is_iota_from_one(a));
  }
  { // The engine can be a const lvalue.
    std::array<unsigned, 3> a{};
    unsigned next = 1;
    const ConstEngine g{&next};
    std::ranges::generate_random(a, g);
    assert(is_iota_from_one(a));
    std::ranges::generate_random(a.begin(), a.end(), g);
    assert((a == std::array<unsigned, 3>{4, 5, 6}));
  }
  { // The engine state is updated, so later calls continue the sequence.
    std::array<unsigned, 3> a{};
    std::array<unsigned, 3> b{};
    SpanMemberEngine g;
    std::ranges::generate_random(a, g);
    std::ranges::generate_random(b, g);
    assert((a == std::array<unsigned, 3>{1, 2, 3}));
    assert((b == std::array<unsigned, 3>{4, 5, 6}));
  }
}

// A forward iterator with a narrow difference type, and a sized sentinel for it. The number of elements must not be
// narrowed to the difference type, which can be smaller than the buffer used with span members.
template <class T, class Diff>
struct NarrowDifferenceIterator {
  using value_type      = T;
  using difference_type = Diff;

  constexpr T& operator*() const { return *ptr; }
  constexpr NarrowDifferenceIterator& operator++() {
    ++ptr;
    return *this;
  }
  constexpr NarrowDifferenceIterator operator++(int) {
    auto tmp = *this;
    ++ptr;
    return tmp;
  }
  friend constexpr bool operator==(const NarrowDifferenceIterator&, const NarrowDifferenceIterator&) = default;

  T* ptr = nullptr;
};

template <class T, class Diff>
struct NarrowDifferenceSentinel {
  using Iter = NarrowDifferenceIterator<T, Diff>;

  friend constexpr bool operator==(const Iter& i, const NarrowDifferenceSentinel& s) { return i.ptr == s.ptr; }
  friend constexpr Diff operator-(const NarrowDifferenceSentinel& s, const Iter& i) {
    return static_cast<Diff>(s.ptr - i.ptr);
  }
  friend constexpr Diff operator-(const Iter& i, const NarrowDifferenceSentinel& s) {
    return static_cast<Diff>(i.ptr - s.ptr);
  }

  T* ptr = nullptr;
};

// An output iterator whose reference type can only be assigned from rvalues. output_range<R, T> only guarantees that
// rvalues of T can be written.
template <class T>
struct RvalueOnlyReference {
  constexpr const RvalueOnlyReference& operator=(T&& x) const {
    *ptr = x;
    return *this;
  }
  const RvalueOnlyReference& operator=(const T&) const = delete;

  T* ptr;
};

template <class T>
struct RvalueOnlyOutputIterator {
  using difference_type = std::ptrdiff_t;

  constexpr RvalueOnlyReference<T> operator*() const { return {ptr}; }
  constexpr RvalueOnlyOutputIterator& operator++() {
    ++ptr;
    return *this;
  }
  constexpr RvalueOnlyOutputIterator operator++(int) {
    auto tmp = *this;
    ++ptr;
    return tmp;
  }

  T* ptr;
};

template <class T>
struct RvalueOnlyOutputSentinel {
  using Iter = RvalueOnlyOutputIterator<T>;

  friend constexpr bool operator==(const Iter& i, const RvalueOnlyOutputSentinel& s) { return i.ptr == s.ptr; }
  friend constexpr std::ptrdiff_t operator-(const RvalueOnlyOutputSentinel& s, const Iter& i) { return s.ptr - i.ptr; }
  friend constexpr std::ptrdiff_t operator-(const Iter& i, const RvalueOnlyOutputSentinel& s) { return i.ptr - s.ptr; }

  T* ptr = nullptr;
};

static_assert(std::forward_iterator<NarrowDifferenceIterator<unsigned, signed char>>);
static_assert(std::sized_sentinel_for<NarrowDifferenceSentinel<unsigned, signed char>,
                                      NarrowDifferenceIterator<unsigned, signed char>>);
static_assert(std::output_iterator<RvalueOnlyOutputIterator<unsigned>, unsigned>);
static_assert(!std::indirectly_writable<RvalueOnlyOutputIterator<unsigned>, unsigned&>);
static_assert(std::sized_sentinel_for<RvalueOnlyOutputSentinel<unsigned>, RvalueOnlyOutputIterator<unsigned>>);

// The member functions can only be called on lvalues. [alg.rand.generate]/1.1 and /1.2 call them on g, which is an
// lvalue even if the engine was passed as an rvalue.
struct LvalueOnlyRangeMemberEngine : CountingEngine {
  template <class R>
  constexpr void generate_random(R&& r) & {
    ++member_calls;
    for (auto&& x : r)
      x = next++;
  }
  template <class R>
  void generate_random(R&&) && = delete;

  int member_calls = 0;
};

// The span member only accepts lvalue spans. [alg.rand.generate]/1.2 passes "an object s".
struct LvalueOnlySpanMemberEngine : CountingEngine {
  template <std::size_t Extent>
  constexpr void generate_random(std::span<result_type, Extent>& s) & {
    ++member_calls;
    for (auto& x : s)
      x = next++;
  }
  template <std::size_t Extent>
  void generate_random(std::span<result_type, Extent>&) && = delete;

  int member_calls = 0;
};

// Only accepts ranges passed as rvalues. [alg.rand.generate]/1.1 forwards the range.
struct RvalueRangeMemberEngine : CountingEngine {
  template <class R>
    requires(!std::is_lvalue_reference_v<R>)
  constexpr void generate_random(R&& r) {
    ++member_calls;
    for (auto&& x : r)
      x = next++;
  }

  int member_calls = 0;
};

constexpr void test_narrow_difference_type() {
  using Iter = NarrowDifferenceIterator<unsigned, signed char>;
  using Sent = NarrowDifferenceSentinel<unsigned, signed char>;
  { // (iterator, sentinel) overload
    std::array<unsigned, 100> a{};
    SpanMemberEngine g;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter{a.data()}, Sent{a.data() + 100}, g);
    assert(result.ptr == a.data() + 100);
    assert(is_iota_from_one(a));
    assert(g.calls + g.elements_generated == 100);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // (range) overload
    std::array<unsigned, 127> a{};
    SpanMemberEngine g;
    auto range                               = std::ranges::subrange(Iter{a.data()}, Sent{a.data() + 127});
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(range, g);
    assert(result.ptr == a.data() + 127);
    assert(is_iota_from_one(a));
    assert(g.calls + g.elements_generated == 127);
    LIBCPP_ASSERT(g.calls == 0);
  }
}

constexpr void test_value_categories() {
  { // The range member is called on an lvalue, even if the engine is passed as an rvalue. The range is not sized, so
    // the member can only be used through [alg.rand.generate]/1.1.
    using Iter = forward_iterator<unsigned*>;
    std::array<unsigned, 5> a{};
    LvalueOnlyRangeMemberEngine g;
    auto range = std::ranges::subrange(Iter(a.data()), sentinel_wrapper<Iter>(Iter(a.data() + a.size())));
    static_assert(!std::ranges::sized_range<decltype(range)>);
    std::ranges::generate_random(range, std::move(g));
    assert(is_iota_from_one(a));
    assert(g.member_calls == 1);
    assert(g.calls == 0);
  }
  { // The span member is called on an lvalue engine with an lvalue span. Contiguous range of the result type.
    std::array<unsigned, 5> a{};
    LvalueOnlySpanMemberEngine g;
    std::ranges::generate_random(a, std::move(g));
    assert(is_iota_from_one(a));
    LIBCPP_ASSERT(g.member_calls == 1);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // Same, for a range of a different value type.
    std::array<unsigned long long, 5> a{};
    LvalueOnlySpanMemberEngine g;
    std::ranges::generate_random(a.begin(), a.end(), std::move(g));
    assert(is_iota_from_one(a));
    LIBCPP_ASSERT(g.member_calls == 1);
    LIBCPP_ASSERT(g.calls == 0);
  }
  { // The range is forwarded to the member.
    std::array<unsigned, 5> a{};
    RvalueRangeMemberEngine g;
    std::ranges::generate_random(std::span<unsigned>(a), g);
    assert(g.member_calls == 1);
    assert(g.calls == 0);

    // An lvalue range cannot be passed to the member, so g() is called for each element.
    std::span<unsigned> s(a);
    std::ranges::generate_random(s, g);
    assert(g.member_calls == 1);
    assert(g.calls == 5);

    // The (iterator, sentinel) overload passes a subrange prvalue.
    std::ranges::generate_random(a.begin(), a.end(), g);
    assert(g.member_calls == 2);
    assert(g.calls == 5);
    assert((a == std::array<unsigned, 5>{11, 12, 13, 14, 15}));
  }
  { // Values are written to the range as rvalues, with and without the span member.
    std::array<unsigned, 300> a{};
    SpanMemberEngine g;
    std::same_as<RvalueOnlyOutputIterator<unsigned>> decltype(auto) result = std::ranges::generate_random(
        RvalueOnlyOutputIterator<unsigned>{a.data()}, RvalueOnlyOutputSentinel<unsigned>{a.data() + 300}, g);
    assert(result.ptr == a.data() + 300);
    assert(is_iota_from_one(a));
    LIBCPP_ASSERT(g.calls == 0);

    std::array<unsigned, 5> b{};
    CountingEngine g2;
    std::ranges::generate_random(
        RvalueOnlyOutputIterator<unsigned>{b.data()}, RvalueOnlyOutputSentinel<unsigned>{b.data() + 5}, g2);
    assert(is_iota_from_one(b));
    assert(g2.calls == 5);
  }
}

constexpr bool test() {
  test_iterators();
  test_ranges();
  test_narrow_difference_type();
  test_value_categories();
  return true;
}

// The effects must be equivalent to ranges::generate(r, ref(g)) ([alg.rand.generate]/3), so the values must be the
// same as the ones produced by calling the engine directly.
template <class Engine, class Container>
void test_standard_engine() {
  Engine expected_engine;
  Container expected(1000);
  for (auto&& x : expected)
    x = expected_engine();

  {
    Engine g;
    Container c(1000);
    std::same_as<typename Container::iterator> decltype(auto) result = std::ranges::generate_random(c, g);
    assert(result == c.end());
    assert(c == expected);
    assert(g == expected_engine);
  }
  {
    Engine g;
    Container c(1000);
    std::same_as<typename Container::iterator> decltype(auto) result =
        std::ranges::generate_random(c.begin(), c.end(), g);
    assert(result == c.end());
    assert(c == expected);
    assert(g == expected_engine);
  }
  {
    Container c(1000);
    std::ranges::generate_random(c, Engine());
    assert(c == expected);
  }
}

template <class Engine>
void test_standard_engine() {
  using T = typename Engine::result_type;
  test_standard_engine<Engine, std::vector<T>>();
  test_standard_engine<Engine, std::deque<T>>();
  test_standard_engine<Engine, std::list<T>>();
  test_standard_engine<Engine, std::vector<long double>>();
}

void test_standard_engines() {
  test_standard_engine<std::minstd_rand0>();
  test_standard_engine<std::minstd_rand>();
  test_standard_engine<std::mt19937>();
  test_standard_engine<std::mt19937_64>();
  test_standard_engine<std::ranlux24_base>();
  test_standard_engine<std::ranlux48_base>();
  test_standard_engine<std::ranlux24>();
  test_standard_engine<std::ranlux48>();
  test_standard_engine<std::knuth_b>();
  test_standard_engine<std::default_random_engine>();
  test_standard_engine<std::independent_bits_engine<std::mt19937, 12, unsigned short>>();
}

int main(int, char**) {
  test();
  static_assert(test());

  test_standard_engines();

  return 0;
}
