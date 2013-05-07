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

// Test unique_ptr(pointer) ctor

#include <memory>
#include <cassert>

// unique_ptr(pointer, deleter()) only requires MoveConstructible deleter

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

int main()
{
    {
    A* p = new A;
    assert(A::count == 1);
    std::unique_ptr<A, Deleter<A> > s(p, Deleter<A>());
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
    }
    assert(A::count == 0);
}
