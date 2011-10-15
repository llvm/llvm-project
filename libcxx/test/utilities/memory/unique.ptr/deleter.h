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

// Example move-only deleter

#ifndef DELETER_H
#define DELETER_H

#include <type_traits>
#include <cassert>

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

    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}
    ~Deleter() {assert(state_ >= 0); state_ = -1;}

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

template <class T>
class Deleter<T[]>
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

    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}
    ~Deleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};

template <class T>
void
swap(Deleter<T>& x, Deleter<T>& y)
{
    Deleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}

template <class T>
class CDeleter
{
    int state_;

public:

    CDeleter() : state_(0) {}
    explicit CDeleter(int s) : state_(s) {}
    ~CDeleter() {assert(state_ >= 0); state_ = -1;}

    template <class U>
        CDeleter(const CDeleter<U>& d)
            : state_(d.state()) {}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};

template <class T>
class CDeleter<T[]>
{
    int state_;

public:

    CDeleter() : state_(0) {}
    explicit CDeleter(int s) : state_(s) {}
    ~CDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};

template <class T>
void
swap(CDeleter<T>& x, CDeleter<T>& y)
{
    CDeleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}

#endif  // DELETER_H
