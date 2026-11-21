//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <random>

// template<class R, class G, class D>
//   requires output_range<R, invoke_result_t<D&, G&>> && invocable<D&, G&> &&
//            uniform_random_bit_generator<remove_cvref_t<G>> &&
//            is_arithmetic_v<invoke_result_t<D&, G&>>
//   constexpr borrowed_iterator_t<R> ranges::generate_random(R&& r, G&& g, D&& d);
//
// template<class G, class D, output_iterator<invoke_result_t<D&, G&>> O, sentinel_for<O> S>
//   requires invocable<D&, G&> && uniform_random_bit_generator<remove_cvref_t<G>> &&
//            is_arithmetic_v<invoke_result_t<D&, G&>>
//   constexpr O ranges::generate_random(O first, S last, G&& g, D&& d);

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <list>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

// A uniform random bit generator that produces 1, 2, 3, ... It also has a generate_random member, which must not be
// used by the distribution overloads.
struct Engine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }

  constexpr result_type operator()() {
    ++calls;
    return next++;
  }

  template <class R>
  constexpr void generate_random(R&&) {
    ++member_calls;
  }

  result_type next = 1;
  int calls        = 0;
  int member_calls = 0;
};
static_assert(std::uniform_random_bit_generator<Engine>);

// A distribution without a generate_random member. It returns twice the value produced by the engine.
struct DoublingDist {
  template <class G>
  constexpr long operator()(G& g) {
    ++calls;
    return 2 * static_cast<long>(g());
  }

  int calls = 0;
};

// Has a generate_random member that accepts any range ([alg.rand.generate]/5.1).
struct RangeMemberDist : DoublingDist {
  template <class R, class G>
  constexpr void generate_random(R&& r, G& g) {
    ++member_calls;
    for (auto&& x : r)
      x = 2 * static_cast<long>(g());
  }

  int member_calls = 0;
};

// Has a generate_random member that only accepts spans of the result type ([alg.rand.generate]/5.2).
struct SpanMemberDist : DoublingDist {
  template <std::size_t Extent, class G>
  constexpr void generate_random(std::span<long, Extent> s, G& g) {
    ++member_calls;
    elements_generated += static_cast<int>(s.size());
    for (auto& x : s)
      x = 2 * static_cast<long>(g());
  }

  int member_calls       = 0;
  int elements_generated = 0;
};

// operator() is const, so the distribution can be passed as a const lvalue.
struct ConstDist {
  template <class G>
  constexpr long operator()(G& g) const {
    return 2 * static_cast<long>(g());
  }
};

template <class Range>
constexpr bool is_even_from_two(const Range& r) {
  long expected = 2;
  for (auto&& x : r) {
    if (x != expected)
      return false;
    expected += 2;
  }
  return true;
}

template <class Iter, class Sent>
constexpr void test_no_member() {
  { // (iterator, sentinel) overload
    std::array<long, 5> a{};
    Engine g;
    DoublingDist d;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.calls == 5);
    assert(g.member_calls == 0);
  }
  { // (range) overload
    std::array<long, 5> a{};
    Engine g;
    DoublingDist d;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(std::move(range), g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.calls == 5);
    assert(g.member_calls == 0);
  }
  { // empty range
    std::array<long, 0> a{};
    Engine g;
    DoublingDist d;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data())), g, d);
    assert(base(result) == a.data());
    assert(d.calls == 0);
    assert(g.calls == 0);
  }
}

// The member must be called exactly once, even for an empty range ([alg.rand.generate]/5.1).
template <class Iter, class Sent>
constexpr void test_range_member() {
  { // (iterator, sentinel) overload
    std::array<long, 5> a{};
    Engine g;
    RangeMemberDist d;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.member_calls == 1);
    assert(d.calls == 0);
  }
  { // (range) overload
    std::array<long, 5> a{};
    Engine g;
    RangeMemberDist d;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(range, g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.member_calls == 1);
    assert(d.calls == 0);
  }
  { // empty range
    std::array<long, 0> a{};
    Engine g;
    RangeMemberDist d;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data())), g, d);
    assert(base(result) == a.data());
    assert(d.member_calls == 1);
    assert(d.calls == 0);
  }
}

// For sized ranges, the span member may be used ([alg.rand.generate]/5.2); libc++ does so and never calls d(g).
// For ranges that are not sized, d(g) must be called once per element ([alg.rand.generate]/5.3).
template <class Iter, class Sent>
constexpr void test_span_member() {
  constexpr bool sized = std::sized_sentinel_for<Sent, Iter>;
  { // (iterator, sentinel) overload
    std::array<long, 5> a{};
    Engine g;
    SpanMemberDist d;
    std::same_as<Iter> decltype(auto) result =
        std::ranges::generate_random(Iter(a.data()), Sent(Iter(a.data() + a.size())), g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.calls + d.elements_generated == 5);
    if constexpr (sized) {
      LIBCPP_ASSERT(d.calls == 0);
    } else {
      assert(d.member_calls == 0);
    }
  }
  { // (range) overload
    std::array<long, 5> a{};
    Engine g;
    SpanMemberDist d;
    auto range                               = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + a.size())));
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(std::move(range), g, d);
    assert(base(result) == a.data() + a.size());
    assert(is_even_from_two(a));
    assert(d.calls + d.elements_generated == 5);
    if constexpr (sized) {
      LIBCPP_ASSERT(d.calls == 0);
    } else {
      assert(d.member_calls == 0);
    }
  }
}

template <class Iter>
constexpr void test_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>) {
    test_no_member<Iter, Iter>();
    test_span_member<Iter, Iter>();
  }
  test_no_member<Iter, sentinel_wrapper<Iter>>();
  test_no_member<Iter, sized_sentinel<Iter>>();
  test_span_member<Iter, sentinel_wrapper<Iter>>();
  test_span_member<Iter, sized_sentinel<Iter>>();

  // See generate_random.pass.cpp for why move-only iterators are not tested with a range member.
  if constexpr (std::copyable<Iter>) {
    if constexpr (std::sentinel_for<Iter, Iter>)
      test_range_member<Iter, Iter>();
    test_range_member<Iter, sentinel_wrapper<Iter>>();
    test_range_member<Iter, sized_sentinel<Iter>>();
  }
}

constexpr void test_iterators() {
  test_iter<cpp17_output_iterator<long*>>();
  test_iter<cpp20_output_iterator<long*>>();
  test_iter<cpp17_input_iterator<long*>>();
  test_iter<forward_iterator<long*>>();
  test_iter<bidirectional_iterator<long*>>();
  test_iter<random_access_iterator<long*>>();
  test_iter<contiguous_iterator<long*>>();
  test_iter<long*>();
}

constexpr void test_ranges() {
  { // Non-borrowed ranges passed as rvalues return ranges::dangling.
    Engine g;
    DoublingDist d;
    std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::generate_random(std::array<long, 3>{}, g, d);
    (void)result;
    assert(d.calls == 3);
  }
  { // A span is passed to the member function as is ([alg.rand.generate]/5.1).
    std::array<long, 4> a{};
    Engine g;
    SpanMemberDist d;
    std::span<long> s(a);
    std::same_as<std::span<long>::iterator> decltype(auto) result = std::ranges::generate_random(s, g, d);
    assert(result == s.end());
    assert(is_even_from_two(a));
    assert(d.member_calls == 1);
    assert(d.calls == 0);
  }
  { // Different value type and more elements than fit in one chunk.
    std::vector<double> v(1000);
    Engine g;
    SpanMemberDist d;
    std::same_as<std::vector<double>::iterator> decltype(auto) result = std::ranges::generate_random(v, g, d);
    assert(result == v.end());
    assert(is_even_from_two(v));
    assert(d.calls + d.elements_generated == 1000);
    LIBCPP_ASSERT(d.member_calls >= 1);
    LIBCPP_ASSERT(d.calls == 0);
  }
  { // The engine and the distribution can be rvalues.
    std::array<long, 3> a{};
    std::ranges::generate_random(a, Engine{}, DoublingDist{});
    assert(is_even_from_two(a));
    std::ranges::generate_random(a.begin(), a.end(), Engine{}, SpanMemberDist{});
    assert(is_even_from_two(a));
  }
  { // The distribution can be a const lvalue.
    std::array<long, 3> a{};
    Engine g;
    const ConstDist d;
    std::ranges::generate_random(a, g, d);
    assert(is_even_from_two(a));
    std::ranges::generate_random(a.begin(), a.end(), g, d);
    assert((a == std::array<long, 3>{8, 10, 12}));
  }
  { // The distribution is called with std::invoke. A pointer to member function of the engine works.
    std::array<unsigned, 3> a{};
    Engine g;
    std::ranges::generate_random(a, g, &Engine::operator());
    assert((a == std::array<unsigned, 3>{1, 2, 3}));
    std::ranges::generate_random(a.begin(), a.end(), g, &Engine::operator());
    assert((a == std::array<unsigned, 3>{4, 5, 6}));
    assert(g.member_calls == 0);
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

static_assert(std::forward_iterator<NarrowDifferenceIterator<long, signed char>>);
static_assert(
    std::sized_sentinel_for<NarrowDifferenceSentinel<long, signed char>, NarrowDifferenceIterator<long, signed char>>);
static_assert(std::output_iterator<RvalueOnlyOutputIterator<long>, long>);
static_assert(!std::indirectly_writable<RvalueOnlyOutputIterator<long>, long&>);

// The member functions can only be called on lvalues, and they only accept lvalue engines. [alg.rand.generate]/5.1
// and /5.2 call them on d with g, which are lvalues even if they were passed as rvalues.
struct LvalueOnlyRangeMemberDist : DoublingDist {
  template <class R, class G>
  constexpr void generate_random(R&& r, G& g) & {
    ++member_calls;
    for (auto&& x : r)
      x = 2 * static_cast<long>(g());
  }
  template <class R, class G>
  void generate_random(R&&, G&) && = delete;

  int member_calls = 0;
};

struct LvalueOnlySpanMemberDist : DoublingDist {
  template <std::size_t Extent, class G>
  constexpr void generate_random(std::span<long, Extent>& s, G& g) & {
    ++member_calls;
    for (auto& x : s)
      x = 2 * static_cast<long>(g());
  }
  template <std::size_t Extent, class G>
  void generate_random(std::span<long, Extent>&, G&) && = delete;

  int member_calls = 0;
};

constexpr void test_narrow_difference_type() {
  using Iter = NarrowDifferenceIterator<long, signed char>;
  using Sent = NarrowDifferenceSentinel<long, signed char>;
  { // (iterator, sentinel) overload
    std::array<long, 100> a{};
    Engine g;
    SpanMemberDist d;
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(Iter{a.data()}, Sent{a.data() + 100}, g, d);
    assert(result.ptr == a.data() + 100);
    assert(is_even_from_two(a));
    assert(d.calls + d.elements_generated == 100);
    LIBCPP_ASSERT(d.calls == 0);
  }
  { // (range) overload
    std::array<long, 127> a{};
    Engine g;
    SpanMemberDist d;
    auto range                               = std::ranges::subrange(Iter{a.data()}, Sent{a.data() + 127});
    std::same_as<Iter> decltype(auto) result = std::ranges::generate_random(range, g, d);
    assert(result.ptr == a.data() + 127);
    assert(is_even_from_two(a));
    assert(d.calls + d.elements_generated == 127);
    LIBCPP_ASSERT(d.calls == 0);
  }
}

constexpr void test_value_categories() {
  { // The range member is called on an lvalue distribution with an lvalue engine. The range is not sized, so the
    // member can only be used through [alg.rand.generate]/5.1.
    using Iter = forward_iterator<long*>;
    std::array<long, 5> a{};
    Engine g;
    LvalueOnlyRangeMemberDist d;
    auto range = std::ranges::subrange(Iter(a.data()), sentinel_wrapper<Iter>(Iter(a.data() + a.size())));
    static_assert(!std::ranges::sized_range<decltype(range)>);
    std::ranges::generate_random(range, std::move(g), std::move(d));
    assert(is_even_from_two(a));
    assert(d.member_calls == 1);
    assert(d.calls == 0);
  }
  { // The span member is called on an lvalue distribution with an lvalue span and an lvalue engine.
    std::array<long, 5> a{};
    Engine g;
    LvalueOnlySpanMemberDist d;
    std::ranges::generate_random(a, std::move(g), std::move(d));
    assert(is_even_from_two(a));
    LIBCPP_ASSERT(d.member_calls == 1);
    LIBCPP_ASSERT(d.calls == 0);
  }
  { // Same, for a range of a different value type.
    std::array<double, 5> a{};
    Engine g;
    LvalueOnlySpanMemberDist d;
    std::ranges::generate_random(a.begin(), a.end(), std::move(g), std::move(d));
    assert(is_even_from_two(a));
    LIBCPP_ASSERT(d.member_calls == 1);
    LIBCPP_ASSERT(d.calls == 0);
  }
  { // Values are written to the range as rvalues, with and without the span member.
    std::array<long, 300> a{};
    Engine g;
    SpanMemberDist d;
    std::same_as<RvalueOnlyOutputIterator<long>> decltype(auto) result = std::ranges::generate_random(
        RvalueOnlyOutputIterator<long>{a.data()}, RvalueOnlyOutputSentinel<long>{a.data() + 300}, g, d);
    assert(result.ptr == a.data() + 300);
    assert(is_even_from_two(a));
    LIBCPP_ASSERT(d.calls == 0);

    std::array<long, 5> b{};
    Engine g2;
    DoublingDist d2;
    std::ranges::generate_random(
        RvalueOnlyOutputIterator<long>{b.data()}, RvalueOnlyOutputSentinel<long>{b.data() + 5}, g2, d2);
    assert(is_even_from_two(b));
    assert(d2.calls == 5);
  }
}

constexpr bool test() {
  test_iterators();
  test_ranges();
  test_narrow_difference_type();
  test_value_categories();
  return true;
}

// Standard distributions have no generate_random member, so the values are the ones produced by d(g). The standard
// only requires the same distribution of values ([alg.rand.generate]/7), so comparing the exact values is
// libc++-specific.
template <class Dist, class Container, class... Args>
void test_standard_distribution(Args... args) {
  std::minstd_rand expected_engine;
  Dist expected_dist(args...);
  Container expected(1000);
  for (auto&& x : expected)
    x = expected_dist(expected_engine);

  {
    std::minstd_rand g;
    Dist d(args...);
    Container c(1000);
    std::same_as<typename Container::iterator> decltype(auto) result = std::ranges::generate_random(c, g, d);
    assert(result == c.end());
    LIBCPP_ASSERT(c == expected);
    LIBCPP_ASSERT(g == expected_engine);
    for (auto x : c)
      assert(d.min() <= x && x <= d.max());
  }
  {
    std::minstd_rand g;
    Dist d(args...);
    Container c(1000);
    std::same_as<typename Container::iterator> decltype(auto) result =
        std::ranges::generate_random(c.begin(), c.end(), g, d);
    assert(result == c.end());
    LIBCPP_ASSERT(c == expected);
    LIBCPP_ASSERT(g == expected_engine);
  }
  {
    Container c(1000);
    std::ranges::generate_random(c, std::minstd_rand(), Dist(args...));
    LIBCPP_ASSERT(c == expected);
  }
}

template <class Dist, class... Args>
void test_standard_distribution_in_containers(Args... args) {
  using T = typename Dist::result_type;
  test_standard_distribution<Dist, std::vector<T>>(args...);
  test_standard_distribution<Dist, std::deque<T>>(args...);
  test_standard_distribution<Dist, std::list<T>>(args...);
}

void test_standard_distributions() {
  test_standard_distribution_in_containers<std::uniform_int_distribution<int>>(1, 6);
  test_standard_distribution_in_containers<std::uniform_int_distribution<unsigned long long>>();
  test_standard_distribution_in_containers<std::uniform_real_distribution<double>>(-1.0, 1.0);
  test_standard_distribution_in_containers<std::uniform_real_distribution<float>>();
  test_standard_distribution_in_containers<std::bernoulli_distribution>(0.25);
  test_standard_distribution_in_containers<std::binomial_distribution<int>>(10, 0.5);
  test_standard_distribution_in_containers<std::geometric_distribution<int>>(0.3);
  test_standard_distribution_in_containers<std::poisson_distribution<int>>(4.0);
  test_standard_distribution_in_containers<std::exponential_distribution<double>>(2.0);
  test_standard_distribution_in_containers<std::normal_distribution<double>>(0.0, 1.0);
  test_standard_distribution_in_containers<std::lognormal_distribution<double>>();
  test_standard_distribution_in_containers<std::student_t_distribution<double>>(3.0);
  test_standard_distribution_in_containers<std::discrete_distribution<int>>(std::initializer_list<double>{1, 2, 3});

  { // Distribution result converted to a different value type.
    std::minstd_rand g;
    std::uniform_int_distribution<int> d(1, 6);
    std::vector<double> v(100);
    std::ranges::generate_random(v, g, d);
    for (double x : v)
      assert(x >= 1.0 && x <= 6.0);
  }
}

int main(int, char**) {
  test();
  static_assert(test());

  test_standard_distributions();

  return 0;
}
