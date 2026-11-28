//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <memory>

// shared_ptr

// template <class T>
// bool
// atomic_compare_exchange_strong_explicit(shared_ptr<T>* p, shared_ptr<T>* v,
//                                         shared_ptr<T> w, memory_order success,
//                                         memory_order failure);

// UNSUPPORTED: c++03

#include <memory>
#include <utility>

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::shared_ptr<int> p(new int(4));
        std::shared_ptr<int> v(new int(3));
        std::shared_ptr<int> w(new int(2));
        bool b = std::atomic_compare_exchange_strong_explicit(&p, &v, w,
                                                              std::memory_order_seq_cst,
                                                              std::memory_order_seq_cst);
        assert(b == false);
        assert(*p == 4);
        assert(*v == 4);
        assert(*w == 2);
    }
    {
        std::shared_ptr<int> p(new int(4));
        std::shared_ptr<int> v = p;
        std::shared_ptr<int> w(new int(2));
        bool b = std::atomic_compare_exchange_strong_explicit(&p, &v, w,
                                                              std::memory_order_seq_cst,
                                                              std::memory_order_seq_cst);
        assert(b == true);
        assert(*p == 2);
        assert(*v == 4);
        assert(*w == 2);
    }
    {
      // Aliases with the same ownership but different stored pointers are not equivalent.
      std::shared_ptr<std::pair<int, int> > owner = std::make_shared<std::pair<int, int> >(7, 11);
      std::shared_ptr<int> p(owner, &owner->first);
      std::shared_ptr<int> v(owner, &owner->second);
      std::shared_ptr<int> w(new int(2));
      bool b =
          std::atomic_compare_exchange_strong_explicit(&p, &v, w, std::memory_order_seq_cst, std::memory_order_seq_cst);
      assert(b == false);
      assert(p.get() == &owner->first);
      assert(v.get() == &owner->first); // expected was updated with p
      assert(*w == 2);
    }
    {
      // Aliases storing the same pointer and sharing ownership are equivalent.
      std::shared_ptr<std::pair<int, int> > owner = std::make_shared<std::pair<int, int> >(7, 11);
      std::shared_ptr<int> p(owner, &owner->first);
      std::shared_ptr<int> v(owner, &owner->first);
      std::shared_ptr<int> w(new int(2));
      bool b =
          std::atomic_compare_exchange_strong_explicit(&p, &v, w, std::memory_order_seq_cst, std::memory_order_seq_cst);
      assert(b == true);
      assert(*p == 2);
      assert(v.get() == &owner->first);
      assert(*w == 2);
    }
    {
      // An ownerless non-null shared_ptr is not equivalent to an empty one.
      int value = 7;
      std::shared_ptr<int> p(std::shared_ptr<int>(), &value);
      std::shared_ptr<int> v;
      std::shared_ptr<int> w(new int(2));
      bool b =
          std::atomic_compare_exchange_strong_explicit(&p, &v, w, std::memory_order_seq_cst, std::memory_order_seq_cst);
      assert(b == false);
      assert(p.get() == &value);
      assert(v.get() == &value); // expected was updated with p
      assert(*w == 2);
    }

  return 0;
}
