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
// };

#include <memory>
#include <cassert>

struct T
    : public std::enable_shared_from_this<T>
{
};

struct Y : T {};

struct Z : Y {};

int main()
{
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
}
