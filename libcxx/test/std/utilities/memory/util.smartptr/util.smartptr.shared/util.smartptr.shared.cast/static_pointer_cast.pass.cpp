//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class U> shared_ptr<T> static_pointer_cast(const shared_ptr<U>& r) noexcept;
// template<class T, class U> shared_ptr<T> static_pointer_cast(shared_ptr<U>&& r) noexcept;

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A& other) : B(other) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    {
        const std::shared_ptr<A> pA(new A);
        ASSERT_NOEXCEPT(std::static_pointer_cast<B>(pA));
        std::shared_ptr<B> pB = std::static_pointer_cast<B>(pA);
        assert(pB.get() == pA.get());
        assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
    {
        const std::shared_ptr<B> pA(new A);
        std::shared_ptr<A> pB = std::static_pointer_cast<A>(pA);
        assert(pB.get() == pA.get());
        assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
    {
        const std::shared_ptr<A> pA;
        std::shared_ptr<B> pB = std::static_pointer_cast<B>(pA);
        assert(pB.get() == pA.get());
        assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
    {
        const std::shared_ptr<B> pA;
        std::shared_ptr<A> pB = std::static_pointer_cast<A>(pA);
        assert(pB.get() == pA.get());
        assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
#if TEST_STD_VER > 14
    {
      const std::shared_ptr<A[8]> pA;
      std::shared_ptr<B[8]> pB = std::static_pointer_cast<B[8]>(pA);
      assert(pB.get() == pA.get());
      assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
    {
      const std::shared_ptr<B[8]> pA;
      std::shared_ptr<A[8]> pB = std::static_pointer_cast<A[8]>(pA);
      assert(pB.get() == pA.get());
      assert(!pB.owner_before(pA) && !pA.owner_before(pB));
    }
#endif // TEST_STD_VER > 14
#if TEST_STD_VER > 20
    {
      A* pA_raw = new A;
      std::shared_ptr<A> pA(pA_raw);
      ASSERT_NOEXCEPT(std::static_pointer_cast<B>(std::move(pA)));
      std::shared_ptr<B> pB = std::static_pointer_cast<B>(std::move(pA));
      assert(pA.get() == nullptr);
      assert(pB.get() == pA_raw);
      assert(pB.use_count() == 1);
    }
#endif // TEST_STD_VER > 20

    return 0;
}
