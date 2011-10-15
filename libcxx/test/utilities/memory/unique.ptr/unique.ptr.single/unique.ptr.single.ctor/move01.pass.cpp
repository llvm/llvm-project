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

// Test unique_ptr move ctor

#include <memory>
#include <cassert>

// test move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

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

class CDeleter
{
    int state_;

    CDeleter(CDeleter&);
    CDeleter& operator=(CDeleter&);
public:

    CDeleter() : state_(5) {}

    int state() const {return state_;}
    void set_state(int s) {state_ = s;}

    void operator()(A* p) {delete p;}
};

int main()
{
    {
    std::unique_ptr<A> s(new A);
    A* p = s.get();
    std::unique_ptr<A> s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 1);
    }
    assert(A::count == 0);
    {
    std::unique_ptr<A, Deleter<A> > s(new A);
    A* p = s.get();
    std::unique_ptr<A, Deleter<A> > s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 1);
    assert(s2.get_deleter().state() == 5);
    assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    {
    CDeleter d;
    std::unique_ptr<A, CDeleter&> s(new A, d);
    A* p = s.get();
    std::unique_ptr<A, CDeleter&> s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 1);
    d.set_state(6);
    assert(s2.get_deleter().state() == d.state());
    assert(s.get_deleter().state() ==  d.state());
    }
    assert(A::count == 0);
}
