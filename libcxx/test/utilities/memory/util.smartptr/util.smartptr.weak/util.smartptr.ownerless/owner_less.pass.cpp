//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> struct owner_less;
//
// template <class T>
// struct owner_less<shared_ptr<T> >
//     : binary_function<shared_ptr<T>, shared_ptr<T>, bool>
// {
//     typedef bool result_type;
//     bool operator()(shared_ptr<T> const&, shared_ptr<T> const&) const;
//     bool operator()(shared_ptr<T> const&, weak_ptr<T> const&) const;
//     bool operator()(weak_ptr<T> const&, shared_ptr<T> const&) const;
// };
//
// template <class T>
// struct owner_less<weak_ptr<T> >
//     : binary_function<weak_ptr<T>, weak_ptr<T>, bool>
// {
//     typedef bool result_type;
//     bool operator()(weak_ptr<T> const&, weak_ptr<T> const&) const;
//     bool operator()(shared_ptr<T> const&, weak_ptr<T> const&) const;
//     bool operator()(weak_ptr<T> const&, shared_ptr<T> const&) const;
// };

#include <memory>
#include <cassert>

int main()
{
    const std::shared_ptr<int> p1(new int);
    const std::shared_ptr<int> p2 = p1;
    const std::shared_ptr<int> p3(new int);
    const std::weak_ptr<int> w1(p1);
    const std::weak_ptr<int> w2(p2);
    const std::weak_ptr<int> w3(p3);

    {
    typedef std::owner_less<std::shared_ptr<int> > CS;
    CS cs;

    assert(!cs(p1, p2));
    assert(!cs(p2, p1));
    assert(cs(p1 ,p3) || cs(p3, p1));
    assert(cs(p3, p1) == cs(p3, p2));

    assert(!cs(p1, w2));
    assert(!cs(p2, w1));
    assert(cs(p1, w3) || cs(p3, w1));
    assert(cs(p3, w1) == cs(p3, w2));
    }
    {
    typedef std::owner_less<std::weak_ptr<int> > CS;
    CS cs;

    assert(!cs(w1, w2));
    assert(!cs(w2, w1));
    assert(cs(w1, w3) || cs(w3, w1));
    assert(cs(w3, w1) == cs(w3, w2));

    assert(!cs(w1, p2));
    assert(!cs(w2, p1));
    assert(cs(w1, p3) || cs(w3, p1));
    assert(cs(w3, p1) == cs(w3, p2));
    }
}
