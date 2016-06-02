//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template<class T>
// class enable_shared_from_this
// {
// protected:
//     enable_shared_from_this();
//     enable_shared_from_this(enable_shared_from_this const&);
//     enable_shared_from_this& operator=(enable_shared_from_this const&);
//     ~enable_shared_from_this();
// public:
//     shared_ptr<T> shared_from_this();
//     shared_ptr<T const> shared_from_this() const;
//     weak_ptr<T> weak_from_this() noexcept;                         // C++17
//     weak_ptr<T const> weak_from_this() const noexecpt;             // C++17
// };

#include <memory>
#include <cassert>

#include "test_macros.h"

struct T
    : public std::enable_shared_from_this<T>
{
};

struct Y : T {};

struct Z : Y {};

void nullDeleter(void*) {}

int main()
{
    {  // https://llvm.org/bugs/show_bug.cgi?id=18843
    std::shared_ptr<T const> t1(new T);
    std::shared_ptr<T const> t2(std::make_shared<T>());
    }
    {
    std::shared_ptr<Y> p(new Z);
    std::shared_ptr<T> q = p->shared_from_this();
    assert(p == q);
    assert(!p.owner_before(q) && !q.owner_before(p)); // p and q share ownership
    }
    {
    std::shared_ptr<Y> p = std::make_shared<Z>();
    std::shared_ptr<T> q = p->shared_from_this();
    assert(p == q);
    assert(!p.owner_before(q) && !q.owner_before(p)); // p and q share ownership
    }
    // Test LWG issue 2529. Only reset '__weak_ptr_' when it's already expired.
    // http://cplusplus.github.io/LWG/lwg-active.html#2529.
    // Test two different ways:
    // * Using 'weak_from_this().expired()' in C++17.
    // * Using 'shared_from_this()' in all dialects.
    {

        T* ptr = new T;
        std::shared_ptr<T> s(ptr);
        {
            // Don't re-initialize the "enabled_shared_from_this" base
            // because it already references a non-expired shared_ptr.
            std::shared_ptr<T> s2(ptr, &nullDeleter);
        }
#if TEST_STD_VER > 14
        // The enabled_shared_from_this base should still be referencing
        // the original shared_ptr.
        assert(!ptr->weak_from_this().expired());
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
        {
            try {
                std::shared_ptr<T> new_s = ptr->shared_from_this();
                assert(new_s == s);
            } catch (std::bad_weak_ptr const&) {
                assert(false);
            } catch (...) {
                assert(false);
            }
        }
#endif
    }
    // Test weak_from_this_methods
#if TEST_STD_VER > 14
    {
        T* ptr = new T;
        const T* cptr = ptr;

        static_assert(noexcept(ptr->weak_from_this()), "Operation must be noexcept");
        static_assert(noexcept(cptr->weak_from_this()), "Operation must be noexcept");

        std::weak_ptr<T> my_weak = ptr->weak_from_this();
        assert(my_weak.expired());

        std::weak_ptr<T const> my_const_weak = cptr->weak_from_this();
        assert(my_const_weak.expired());

        // Enable shared_from_this with ptr.
        std::shared_ptr<T> sptr(ptr);
        my_weak = ptr->weak_from_this();
        assert(!my_weak.expired());
        assert(my_weak.lock().get() == ptr);
    }
#endif
}
