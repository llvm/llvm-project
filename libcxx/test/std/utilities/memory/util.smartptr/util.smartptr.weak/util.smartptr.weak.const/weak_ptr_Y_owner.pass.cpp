//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// template<class Y> weak_ptr(const weak_ptr<Y>& r);
// template<class Y> weak_ptr(weak_ptr<Y>&& r);

#include <cassert>
#include <memory>
#include <utility>

struct Base {
  virtual ~Base() {}
};

struct Derived : Base {};

struct VirtualBase {
  virtual ~VirtualBase() {}
};

struct VirtualDerived : virtual VirtualBase {};

template <class T, class U>
bool same_owner(const std::weak_ptr<T>& lhs, const std::weak_ptr<U>& rhs) {
  return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

int main(int, char**) {
  {
    std::shared_ptr<Derived> strong(new Derived);
    std::weak_ptr<Derived> owner(strong);
    strong.reset();
    std::weak_ptr<Derived> empty;
    assert(owner.expired());
    assert(!same_owner(owner, empty));

    std::weak_ptr<Base> copied(owner);
    assert(copied.expired());
    assert(copied.use_count() == 0);
    assert(same_owner(owner, copied));

    std::weak_ptr<Derived> moved_from(owner);
    std::weak_ptr<Base> moved(std::move(moved_from));
    assert(moved.expired());
    assert(moved.use_count() == 0);
    assert(same_owner(owner, moved));
    assert(same_owner(moved_from, empty));
  }
  {
    std::weak_ptr<Derived> empty;
    std::weak_ptr<Base> copied(empty);
    std::weak_ptr<Base> moved(std::move(empty));
    assert(same_owner(copied, moved));
    assert(same_owner(empty, moved));
  }
  {
    std::shared_ptr<VirtualDerived> strong(new VirtualDerived);
    VirtualBase* base = strong.get();
    std::weak_ptr<VirtualDerived> source(strong);
    std::weak_ptr<VirtualBase> copied(source);
    assert(copied.lock().get() == base);
    std::weak_ptr<VirtualBase> moved(std::move(source));
    assert(moved.lock().get() == base);
    std::weak_ptr<VirtualDerived> empty;
    assert(same_owner(source, empty));
  }
  return 0;
}
