//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions
// `check_assertion.h` requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: no-localization

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>
// <numeric>
//
// Check that PSTL algorithms terminate on user-thrown exceptions.

#include <algorithm>
#include <numeric>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

template <class F>
void assert_non_throwing(F f) {
  // We wrap this whole test in EXPECT_STD_TERMINATE because if f() terminates, we want the test to pass,
  // since this signals proper handling of user exceptions in the PSTL.
  EXPECT_STD_TERMINATE([&] {
    bool threw = false;
    try {
      f();
    } catch (...) {
      threw = true;
    }
    // If nothing was thrown, call std::terminate() to pass the EXPECT_STD_TERMINATE assertion.
    // Otherwise, don't call std::terminate() to fail the assertion.
    if (!threw)
      std::terminate();
  });
}

struct ThrowToken {
  void activate() { active_ = true; }
  void deactivate() { active_ = false; }
  bool active() const { return active_; }

private:
  bool active_{false};
};

template <class Func>
struct on_scope_exit {
  explicit on_scope_exit(Func func) : func_(func) {}
  ~on_scope_exit() { func_(); }

private:
  Func func_;
};
template <class Func>
on_scope_exit(Func) -> on_scope_exit<Func>;

int main(int, char**) {
  test_execution_policies([&](auto&& policy) {
    int a[] = {1, 2, 3, 4};
    int b[] = {1, 2, 3};
    int n   = 2;
    int storage[999];
    int val  = 99;
    int init = 1;

    // We generate a certain number of "tokens" and we activate exactly one on each iteration. We then
    // throw in a given operation only when that token is active. That way we check that each argument
    // of the algorithm is handled properly.
    ThrowToken tokens[7];
    for (ThrowToken& t : tokens) {
      t.activate();
      on_scope_exit _([&] { t.deactivate(); });

      auto first1      = util::throw_on_move_iterator(std::begin(a), tokens[0].active() ? 1 : -1);
      auto last1       = util::throw_on_move_iterator(std::end(a), tokens[1].active() ? 1 : -1);
      auto first2      = util::throw_on_move_iterator(std::begin(b), tokens[2].active() ? 1 : -1);
      auto last2       = util::throw_on_move_iterator(std::end(b), tokens[3].active() ? 1 : -1);
      auto dest        = util::throw_on_move_iterator(std::end(storage), tokens[4].active() ? 1 : -1);
      auto maybe_throw = [](ThrowToken const& token, auto f) {
        return [&token, f](auto... args) {
          if (token.active())
            throw 1;
          return f(args...);
        };
      };

      {
        auto pred = maybe_throw(tokens[5], [](int x) -> bool { return x % 2 == 0; });

        // all_of(first, last, pred)
        assert_non_throwing([=, &policy] { (void)std::all_of(policy, std::move(first1), std::move(last1), pred); });

        // any_of(first, last, pred)
        assert_non_throwing([=, &policy] { (void)std::any_of(policy, std::move(first1), std::move(last1), pred); });

        // none_of(first, last, pred)
        assert_non_throwing([=, &policy] { (void)std::none_of(policy, std::move(first1), std::move(last1), pred); });
      }

      {
        // copy(first, last, dest)
        assert_non_throwing([=, &policy] {
          (void)std::copy(policy, std::move(first1), std::move(last1), std::move(dest));
        });

        // copy_n(first, n, dest)
        assert_non_throwing([=, &policy] { (void)std::copy_n(policy, std::move(first1), n, std::move(dest)); });
      }

      {
        auto pred = maybe_throw(tokens[5], [](int x) -> bool { return x % 2 == 0; });

        // count(first, last, val)
        assert_non_throwing([=, &policy] { (void)std::count(policy, std::move(first1), std::move(last1), val); });

        // count_if(first, last, pred)
        assert_non_throwing([=, &policy] { (void)std::count_if(policy, std::move(first1), std::move(last1), pred); });
      }

      {
        auto binary_pred = maybe_throw(tokens[5], [](int x, int y) -> bool { return x == y; });

        // equal(first1, last1, first2)
        assert_non_throwing([=, &policy] {
          (void)std::equal(policy, std::move(first1), std::move(last1), std::move(first2));
        });

        // equal(first1, last1, first2, binary_pred)
        assert_non_throwing([=, &policy] {
          (void)std::equal(policy, std::move(first1), std::move(last1), std::move(first2), binary_pred);
        });

        // equal(first1, last1, first2, last2)
        assert_non_throwing([=, &policy] {
          (void)std::equal(policy, std::move(first1), std::move(last1), std::move(first2), std::move(last2));
        });

        // equal(first1, last1, first2, last2, binary_pred)
        assert_non_throwing([=, &policy] {
          (void)std::equal(
              policy, std::move(first1), std::move(last1), std::move(first2), std::move(last2), binary_pred);
        });
      }

      {
        // fill(first, last, val)
        assert_non_throwing([=, &policy] { (void)std::fill(policy, std::move(first1), std::move(last1), val); });

        // fill_n(first, n, val)
        assert_non_throwing([=, &policy] { (void)std::fill_n(policy, std::move(first1), n, val); });
      }

      {
        auto pred = maybe_throw(tokens[5], [](int x) -> bool { return x % 2 == 0; });

        // find(first, last, val)
        assert_non_throwing([=, &policy] { (void)std::find(policy, std::move(first1), std::move(last1), val); });

        // find_if(first, last, pred)
        assert_non_throwing([=, &policy] { (void)std::find_if(policy, std::move(first1), std::move(last1), pred); });

        // find_if_not(first, last, pred)
        assert_non_throwing([=, &policy] {
          (void)std::find_if_not(policy, std::move(first1), std::move(last1), pred);
        });
      }

      {
        auto func = maybe_throw(tokens[5], [](int) {});

        // for_each(first, last, func)
        assert_non_throwing([=, &policy] { (void)std::for_each(policy, std::move(first1), std::move(last1), func); });

        // for_each_n(first, n, func)
        assert_non_throwing([=, &policy] { (void)std::for_each_n(policy, std::move(first1), n, func); });
      }

      {
        auto gen = maybe_throw(tokens[5], []() -> int { return 42; });

        // generate(first, last, func)
        assert_non_throwing([=, &policy] { (void)std::generate(policy, std::move(first1), std::move(last1), gen); });

        // generate_n(first, n, func)
        assert_non_throwing([=, &policy] { (void)std::generate_n(policy, std::move(first1), n, gen); });
      }

      {
        auto pred = maybe_throw(tokens[5], [](int x) -> bool { return x % 2 == 0; });

        // is_partitioned(first, last, pred)
        assert_non_throwing([=, &policy] {
          (void)std::is_partitioned(policy, std::move(first1), std::move(last1), pred);
        });
      }

      {
        auto compare = maybe_throw(tokens[5], [](int x, int y) -> bool { return x < y; });

        // merge(first1, last1, first2, last2, dest)
        assert_non_throwing([=, &policy] {
          (void)std::merge(
              policy, std::move(first1), std::move(last1), std::move(first2), std::move(last2), std::move(dest));
        });

        // merge(first1, last1, first2, last2, dest, comp)
        assert_non_throwing([=, &policy] {
          (void)std::merge(
              policy,
              std::move(first1),
              std::move(last1),
              std::move(first2),
              std::move(last2),
              std::move(dest),
              compare);
        });
      }

      {
        // move(first, last, dest)
        assert_non_throwing([=, &policy] {
          (void)std::move(policy, std::move(first1), std::move(last1), std::move(dest));
        });
      }

      {
        auto pred = maybe_throw(tokens[5], [](int x) -> bool { return x % 2 == 0; });

        // replace_if(first, last, pred, val)
        assert_non_throwing([=, &policy] {
          (void)std::replace_if(policy, std::move(first1), std::move(last1), pred, val);
        });

        // replace(first, last, val1, val2)
        assert_non_throwing([=, &policy] {
          (void)std::replace(policy, std::move(first1), std::move(last1), val, val);
        });

        // replace_copy_if(first, last, dest, pred, val)
        assert_non_throwing([=, &policy] {
          (void)std::replace_copy_if(policy, std::move(first1), std::move(last1), std::move(dest), pred, val);
        });

        // replace_copy(first, last, dest, val1, val2)
        assert_non_throwing([=, &policy] {
          (void)std::replace_copy(policy, std::move(first1), std::move(last1), std::move(dest), val, val);
        });
      }

      {
        auto mid1 = util::throw_on_move_iterator(std::begin(a) + 2, tokens[5].active() ? 1 : -1);

        // rotate_copy(first, mid, last, dest)
        assert_non_throwing([=, &policy] {
          (void)std::rotate_copy(policy, std::move(first1), std::move(mid1), std::move(last1), std::move(dest));
        });
      }

      {
        auto compare = maybe_throw(tokens[5], [](int x, int y) -> bool { return x < y; });

        // sort(first, last)
        assert_non_throwing([=, &policy] { (void)std::sort(policy, std::move(first1), std::move(last1)); });

        // sort(first, last, comp)
        assert_non_throwing([=, &policy] { (void)std::sort(policy, std::move(first1), std::move(last1), compare); });

        // stable_sort(first, last)
        assert_non_throwing([=, &policy] { (void)std::stable_sort(policy, std::move(first1), std::move(last1)); });

        // stable_sort(first, last, comp)
        assert_non_throwing([=, &policy] {
          (void)std::stable_sort(policy, std::move(first1), std::move(last1), compare);
        });
      }

      {
        auto unary  = maybe_throw(tokens[5], [](int x) -> int { return x * 2; });
        auto binary = maybe_throw(tokens[5], [](int x, int y) -> int { return x * y; });

        // transform(first, last, dest, func)
        assert_non_throwing([=, &policy] {
          (void)std::transform(policy, std::move(first1), std::move(last1), std::move(dest), unary);
        });

        // transform(first1, last1, first2, dest, func)
        assert_non_throwing([=, &policy] {
          (void)std::transform(policy, std::move(first1), std::move(last1), std::move(first2), std::move(dest), binary);
        });
      }

      {
        auto reduction        = maybe_throw(tokens[5], [](int x, int y) -> int { return x + y; });
        auto transform_unary  = maybe_throw(tokens[6], [](int x) -> int { return x * 2; });
        auto transform_binary = maybe_throw(tokens[6], [](int x, int y) -> int { return x * y; });

        // transform_reduce(first1, last1, first2, init)
        assert_non_throwing([=, &policy] {
          (void)std::transform_reduce(policy, std::move(first1), std::move(last1), std::move(first2), init);
        });

        // transform_reduce(first1, last1, init, reduce, transform)
        assert_non_throwing([=, &policy] {
          (void)std::transform_reduce(policy, std::move(first1), std::move(last1), init, reduction, transform_unary);
        });

        // transform_reduce(first1, last1, first2, init, reduce, transform)
        assert_non_throwing([=, &policy] {
          (void)std::transform_reduce(
              policy, std::move(first1), std::move(last1), std::move(first2), init, reduction, transform_binary);
        });
      }

      {
        auto reduction = maybe_throw(tokens[5], [](int x, int y) -> int { return x + y; });

        // reduce(first, last)
        assert_non_throwing([=, &policy] { (void)std::reduce(policy, std::move(first1), std::move(last1)); });

        // reduce(first, last, init)
        assert_non_throwing([=, &policy] { (void)std::reduce(policy, std::move(first1), std::move(last1), init); });

        // reduce(first, last, init, binop)
        assert_non_throwing([=, &policy] {
          (void)std::reduce(policy, std::move(first1), std::move(last1), init, reduction);
        });
      }
    }
  });
}
