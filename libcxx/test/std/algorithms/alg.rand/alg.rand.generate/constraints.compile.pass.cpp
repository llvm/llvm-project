//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <random>

// Check the constraints of ranges::generate_random ([alg.rand.generate]), including the is_arithmetic_v constraint
// on the distribution overloads added by LWG4085.

#include <array>
#include <random>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class R, class G>
concept HasGenerateRandomRange =
    requires(R&& r, G&& g) { std::ranges::generate_random(std::forward<R>(r), std::forward<G>(g)); };

template <class I, class S, class G>
concept HasGenerateRandomIter = requires(I&& i, S&& s, G&& g) {
  std::ranges::generate_random(std::forward<I>(i), std::forward<S>(s), std::forward<G>(g));
};

template <class R, class G, class D>
concept HasGenerateRandomRangeDist = requires(R&& r, G&& g, D&& d) {
  std::ranges::generate_random(std::forward<R>(r), std::forward<G>(g), std::forward<D>(d));
};

template <class I, class S, class G, class D>
concept HasGenerateRandomIterDist = requires(I&& i, S&& s, G&& g, D&& d) {
  std::ranges::generate_random(std::forward<I>(i), std::forward<S>(s), std::forward<G>(g), std::forward<D>(d));
};

struct Engine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }
  result_type operator()();
};
static_assert(std::uniform_random_bit_generator<Engine>);

struct ConstEngine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }
  result_type operator()() const;
};
static_assert(std::uniform_random_bit_generator<ConstEngine>);

// !uniform_random_bit_generator: the result type is signed.
struct SignedEngine {
  using result_type = int;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 100; }
  result_type operator()();
};
static_assert(!std::uniform_random_bit_generator<SignedEngine>);

// !uniform_random_bit_generator: min() is not less than max().
struct MinEqualsMaxEngine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 0; }
  result_type operator()();
};
static_assert(!std::uniform_random_bit_generator<MinEqualsMaxEngine>);

// !uniform_random_bit_generator: no min() and max().
struct NoMinMaxEngine {
  using result_type = unsigned;
  result_type operator()();
};
static_assert(!std::uniform_random_bit_generator<NoMinMaxEngine>);

// !uniform_random_bit_generator: not invocable.
struct NotInvocableEngine {
  using result_type = unsigned;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0u; }
};
static_assert(!std::uniform_random_bit_generator<NotInvocableEngine>);

// Engine overloads.

// (range) overload.
static_assert(HasGenerateRandomRange<std::array<unsigned, 3>&, Engine&>);
static_assert(HasGenerateRandomRange<std::array<unsigned, 3>&, Engine>);
static_assert(HasGenerateRandomRange<std::array<unsigned, 3>, Engine&>); // rvalue range, returns dangling
static_assert(HasGenerateRandomRange<std::array<long, 3>&, Engine&>);
static_assert(HasGenerateRandomRange<std::array<double, 3>&, Engine&>);
static_assert(HasGenerateRandomRange<UncheckedRange<unsigned*>, Engine&>);
static_assert(HasGenerateRandomRange<std::array<unsigned, 3>&, const ConstEngine&>);

// !output_range<R, invoke_result_t<G&>>
static_assert(!HasGenerateRandomRange<const std::array<unsigned, 3>&, Engine&>);
static_assert(!HasGenerateRandomRange<std::array<unsigned*, 3>&, Engine&>);
static_assert(!HasGenerateRandomRange<InputRangeNotInputOrOutputIterator, Engine&>);
static_assert(!HasGenerateRandomRange<OutputRangeNotSentinelSemiregular, Engine&>);
static_assert(!HasGenerateRandomRange<OutputRangeNotSentinelEqualityComparableWith, Engine&>);
static_assert(!HasGenerateRandomRange<int, Engine&>);

// G& is not invocable: operator() is not const.
static_assert(!HasGenerateRandomRange<std::array<unsigned, 3>&, const Engine&>);

// !uniform_random_bit_generator<remove_cvref_t<G>>
static_assert(!HasGenerateRandomRange<std::array<unsigned, 3>&, SignedEngine&>);
static_assert(!HasGenerateRandomRange<std::array<unsigned, 3>&, MinEqualsMaxEngine&>);
static_assert(!HasGenerateRandomRange<std::array<unsigned, 3>&, NoMinMaxEngine&>);
static_assert(!HasGenerateRandomRange<std::array<unsigned, 3>&, NotInvocableEngine&>);

// (iterator, sentinel) overload.
static_assert(HasGenerateRandomIter<unsigned*, unsigned*, Engine&>);
static_assert(HasGenerateRandomIter<unsigned*, unsigned*, Engine>);
static_assert(HasGenerateRandomIter<int*, int*, Engine&>);
static_assert(HasGenerateRandomIter<cpp17_output_iterator<unsigned*>,
                                    sentinel_wrapper<cpp17_output_iterator<unsigned*>>,
                                    Engine&>);
static_assert(HasGenerateRandomIter<cpp20_output_iterator<unsigned*>,
                                    sentinel_wrapper<cpp20_output_iterator<unsigned*>>,
                                    Engine&>);
static_assert(HasGenerateRandomIter<unsigned*, unsigned*, const ConstEngine&>);

// !output_iterator<O, invoke_result_t<G&>>
static_assert(!HasGenerateRandomIter<const unsigned*, const unsigned*, Engine&>);
static_assert(!HasGenerateRandomIter<unsigned**, unsigned**, Engine&>);
static_assert(!HasGenerateRandomIter<cpp20_input_iterator<unsigned*>,
                                     sentinel_wrapper<cpp20_input_iterator<unsigned*>>,
                                     Engine&>);
static_assert(!HasGenerateRandomIter<WeaklyIncrementableNotMovable, WeaklyIncrementableNotMovable, Engine&>);

// !sentinel_for<S, O>
static_assert(!HasGenerateRandomIter<int*, SentinelForNotSemiregular, Engine&>);
static_assert(!HasGenerateRandomIter<int*, SentinelForNotWeaklyEqualityComparableWith, Engine&>);
static_assert(!HasGenerateRandomIter<unsigned*, long*, Engine&>);

// G& is not invocable: operator() is not const.
static_assert(!HasGenerateRandomIter<unsigned*, unsigned*, const Engine&>);

// !uniform_random_bit_generator<remove_cvref_t<G>>
static_assert(!HasGenerateRandomIter<unsigned*, unsigned*, SignedEngine&>);
static_assert(!HasGenerateRandomIter<unsigned*, unsigned*, MinEqualsMaxEngine&>);
static_assert(!HasGenerateRandomIter<unsigned*, unsigned*, NoMinMaxEngine&>);
static_assert(!HasGenerateRandomIter<unsigned*, unsigned*, NotInvocableEngine&>);

// Distribution overloads.

struct Dist {
  template <class G>
  long operator()(G&);
};

struct ConstDist {
  template <class G>
  long operator()(G&) const;
};

struct BoolDist {
  bool operator()(Engine&);
};

// !invocable<D&, G&>: only accepts another engine type.
struct ConstEngineOnlyDist {
  long operator()(ConstEngine&);
};

// !invocable<D&, G&>: only accepts rvalue engines.
struct RvalueEngineDist {
  long operator()(Engine&&);
};

// !is_arithmetic_v<invoke_result_t<D&, G&>> (LWG4085): returns a reference.
struct RefDist {
  long& operator()(Engine&);
};

struct ConstRefDist {
  const long& operator()(Engine&);
};

// !is_arithmetic_v<invoke_result_t<D&, G&>> (LWG4085): returns a class type that converts to an arithmetic type.
struct ConvertibleToLong {
  operator long() const;
};

struct ClassDist {
  ConvertibleToLong operator()(Engine&);
};

// !is_arithmetic_v<invoke_result_t<D&, G&>> (LWG4085): the motivating example of the issue. The result is a reference
// to an object that is assignable but not copy constructible, so a lambda without a trailing return type that
// returned invoke(d, g) would not compile. The constraint rejects it instead of a hard error.
struct NotCopyConstructible {
  NotCopyConstructible(const NotCopyConstructible&)            = delete;
  NotCopyConstructible& operator=(const NotCopyConstructible&) = default;
};

struct NotCopyConstructibleDist {
  NotCopyConstructible& operator()(Engine&);
};
static_assert(std::ranges::output_range<std::array<NotCopyConstructible, 3>&, NotCopyConstructible&>);

// (range) overload.
static_assert(HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, Dist&>);
static_assert(HasGenerateRandomRangeDist<std::array<long, 3>&, Engine, Dist>);
static_assert(HasGenerateRandomRangeDist<std::array<long, 3>, Engine&, Dist&>); // rvalue range, returns dangling
static_assert(HasGenerateRandomRangeDist<std::array<double, 3>&, Engine&, Dist&>);
static_assert(HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, const ConstDist&>);
static_assert(HasGenerateRandomRangeDist<std::array<long, 3>&, const ConstEngine&, Dist&>);
static_assert(HasGenerateRandomRangeDist<std::array<bool, 3>&, Engine&, BoolDist&>);
static_assert(HasGenerateRandomRangeDist<std::array<unsigned, 3>&, Engine&, unsigned (Engine::*)()>);

// !output_range<R, invoke_result_t<D&, G&>>
static_assert(!HasGenerateRandomRangeDist<const std::array<long, 3>&, Engine&, Dist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long*, 3>&, Engine&, Dist&>);
static_assert(!HasGenerateRandomRangeDist<InputRangeNotInputOrOutputIterator, Engine&, Dist&>);
static_assert(!HasGenerateRandomRangeDist<OutputRangeNotSentinelSemiregular, Engine&, Dist&>);

// !invocable<D&, G&>
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, const Dist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, ConstEngineOnlyDist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine, RvalueEngineDist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, Engine&>);

// !uniform_random_bit_generator<remove_cvref_t<G>>
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, SignedEngine&, Dist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, MinEqualsMaxEngine&, Dist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, NoMinMaxEngine&, Dist&>);

// !is_arithmetic_v<invoke_result_t<D&, G&>>
static_assert(std::ranges::output_range<std::array<long, 3>&, long&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, RefDist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, ConstRefDist&>);
static_assert(std::ranges::output_range<std::array<long, 3>&, ConvertibleToLong>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, ClassDist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<NotCopyConstructible, 3>&, Engine&, NotCopyConstructibleDist&>);
static_assert(!HasGenerateRandomRangeDist<std::array<long, 3>&, Engine&, long Engine::*>);

// (iterator, sentinel) overload.
static_assert(HasGenerateRandomIterDist<long*, long*, Engine&, Dist&>);
static_assert(HasGenerateRandomIterDist<long*, long*, Engine, Dist>);
static_assert(HasGenerateRandomIterDist<double*, double*, Engine&, Dist&>);
static_assert(HasGenerateRandomIterDist<cpp20_output_iterator<long*>,
                                        sentinel_wrapper<cpp20_output_iterator<long*>>,
                                        Engine&,
                                        Dist&>);
static_assert(HasGenerateRandomIterDist<long*, long*, Engine&, const ConstDist&>);
static_assert(HasGenerateRandomIterDist<unsigned*, unsigned*, Engine&, unsigned (Engine::*)()>);

// !output_iterator<O, invoke_result_t<D&, G&>>
static_assert(!HasGenerateRandomIterDist<const long*, const long*, Engine&, Dist&>);
static_assert(!HasGenerateRandomIterDist<long**, long**, Engine&, Dist&>);
static_assert(!HasGenerateRandomIterDist<cpp20_input_iterator<long*>,
                                         sentinel_wrapper<cpp20_input_iterator<long*>>,
                                         Engine&,
                                         Dist&>);

// !sentinel_for<S, O>
static_assert(!HasGenerateRandomIterDist<int*, SentinelForNotSemiregular, Engine&, Dist&>);
static_assert(!HasGenerateRandomIterDist<int*, SentinelForNotWeaklyEqualityComparableWith, Engine&, Dist&>);

// !invocable<D&, G&>
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine&, const Dist&>);
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine&, ConstEngineOnlyDist&>);
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine, RvalueEngineDist&>);

// !uniform_random_bit_generator<remove_cvref_t<G>>
static_assert(!HasGenerateRandomIterDist<long*, long*, SignedEngine&, Dist&>);
static_assert(!HasGenerateRandomIterDist<long*, long*, NoMinMaxEngine&, Dist&>);

// !is_arithmetic_v<invoke_result_t<D&, G&>>
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine&, RefDist&>);
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine&, ConstRefDist&>);
static_assert(!HasGenerateRandomIterDist<long*, long*, Engine&, ClassDist&>);
static_assert(
    !HasGenerateRandomIterDist<NotCopyConstructible*, NotCopyConstructible*, Engine&, NotCopyConstructibleDist&>);
