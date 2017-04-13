//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test default unique_ptr<T[]> ctor

// default unique_ptr<T[]> ctor shouldn't require complete type

#include <memory>
#include <cassert>

struct A;

class Deleter {
  int state_;

  Deleter(Deleter&);
  Deleter& operator=(Deleter&);

public:
  Deleter() : state_(5) {}

  int state() const { return state_; }

  void operator()(A* p);
};

void check(int i);

template <class D = std::default_delete<A> >
struct B {
  std::unique_ptr<A[], D> a_;
  B();
  ~B();

  A* get() const { return a_.get(); }
  D& get_deleter() { return a_.get_deleter(); }
};

int main() {
  {
    B<> s;
    assert(s.get() == 0);
  }
  check(0);
  {
    B<Deleter> s;
    assert(s.get() == 0);
    assert(s.get_deleter().state() == 5);
  }
  check(0);
}

struct A {
  static int count;
  A() { ++count; }
  A(const A&) { ++count; }
  ~A() { --count; }
};

int A::count = 0;

void Deleter::operator()(A* p) { delete p; }

void check(int i) { assert(A::count == i); }

template <class D>
B<D>::B() {}

template <class D>
B<D>::~B() {}
