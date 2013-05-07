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

// Test unique_ptr converting move ctor

#include <memory>
#include <cassert>

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Implicit version

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    virtual ~A() {--count;}
};

int A::count = 0;

struct B
    : public A
{
    static int count;
    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

template <class T>
class Deleter
{
    int state_;

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    Deleter(const Deleter&);
    Deleter& operator=(const Deleter&);
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    Deleter(Deleter&);
    Deleter& operator=(Deleter&);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

public:
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    Deleter(Deleter&& r) : state_(r.state_) {r.state_ = 0;}
    Deleter& operator=(Deleter&& r)
    {
        state_ = r.state_;
        r.state_ = 0;
        return *this;
    }
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    operator std::__rv<Deleter>() {return std::__rv<Deleter>(*this);}
    Deleter(std::__rv<Deleter> r) : state_(r->state_) {r->state_ = 0;}
    Deleter& operator=(std::__rv<Deleter> r)
    {
        state_ = r->state_;
        r->state_ = 0;
        return *this;
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

    Deleter() : state_(5) {}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    template <class U>
        Deleter(Deleter<U>&& d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
            : state_(d.state()) {d.set_state(0);}

private:
    template <class U>
        Deleter(const Deleter<U>& d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0);
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    template <class U>
        Deleter(Deleter<U> d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
            : state_(d.state()) {}
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
public:
    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};

int main()
{
    {
    const std::unique_ptr<B, Deleter<B> > s(new B);
    A* p = s.get();
    std::unique_ptr<A, Deleter<A> > s2 = s;
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 1);
    assert(B::count == 1);
    assert(s2.get_deleter().state() == 5);
    assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}
