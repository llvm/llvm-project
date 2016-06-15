//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_ALLOCATOR_H
#define TEST_ALLOCATOR_H

#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <new>
#include <climits>
#include <cassert>

#include "test_macros.h"

class test_alloc_base
{
protected:
    static int time_to_throw;
public:
    static int throw_after;
    static int count;
    static int alloc_count;
};

int test_alloc_base::count = 0;
int test_alloc_base::time_to_throw = 0;
int test_alloc_base::alloc_count = 0;
int test_alloc_base::throw_after = INT_MAX;

template <class T>
class test_allocator
    : public test_alloc_base
{
    int data_;

    template <class U> friend class test_allocator;
public:

    typedef unsigned                                                   size_type;
    typedef int                                                        difference_type;
    typedef T                                                          value_type;
    typedef value_type*                                                pointer;
    typedef const value_type*                                          const_pointer;
    typedef typename std::add_lvalue_reference<value_type>::type       reference;
    typedef typename std::add_lvalue_reference<const value_type>::type const_reference;

    template <class U> struct rebind {typedef test_allocator<U> other;};

    test_allocator() throw() : data_(0) {++count;}
    explicit test_allocator(int i) throw() : data_(i) {++count;}
    test_allocator(const test_allocator& a) throw()
        : data_(a.data_) {++count;}
    template <class U> test_allocator(const test_allocator<U>& a) throw()
        : data_(a.data_) {++count;}
    ~test_allocator() throw() {assert(data_ >= 0); --count; data_ = -1;}
    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}
    pointer allocate(size_type n, const void* = 0)
        {
            assert(data_ >= 0);
            if (time_to_throw >= throw_after) {
#ifndef _LIBCPP_NO_EXCEPTIONS
                throw std::bad_alloc();
#else
                std::terminate();
#endif
            }
            ++time_to_throw;
            ++alloc_count;
            return (pointer)::operator new(n * sizeof(T));
        }
    void deallocate(pointer p, size_type)
        {assert(data_ >= 0); --alloc_count; ::operator delete((void*)p);}
    size_type max_size() const throw()
        {return UINT_MAX / sizeof(T);}
#if TEST_STD_VER < 11
    void construct(pointer p, const T& val)
        {::new(static_cast<void*>(p)) T(val);}
#else
    template <class U> void construct(pointer p, U&& val)
        {::new(static_cast<void*>(p)) T(std::forward<U>(val));}
#endif
    void destroy(pointer p) {p->~T();}
    friend bool operator==(const test_allocator& x, const test_allocator& y)
        {return x.data_ == y.data_;}
    friend bool operator!=(const test_allocator& x, const test_allocator& y)
        {return !(x == y);}
};

template <class T>
class non_default_test_allocator
    : public test_alloc_base
{
    int data_;

    template <class U> friend class non_default_test_allocator;
public:

    typedef unsigned                                                   size_type;
    typedef int                                                        difference_type;
    typedef T                                                          value_type;
    typedef value_type*                                                pointer;
    typedef const value_type*                                          const_pointer;
    typedef typename std::add_lvalue_reference<value_type>::type       reference;
    typedef typename std::add_lvalue_reference<const value_type>::type const_reference;

    template <class U> struct rebind {typedef non_default_test_allocator<U> other;};

//    non_default_test_allocator() throw() : data_(0) {++count;}
    explicit non_default_test_allocator(int i) throw() : data_(i) {++count;}
    non_default_test_allocator(const non_default_test_allocator& a) throw()
        : data_(a.data_) {++count;}
    template <class U> non_default_test_allocator(const non_default_test_allocator<U>& a) throw()
        : data_(a.data_) {++count;}
    ~non_default_test_allocator() throw() {assert(data_ >= 0); --count; data_ = -1;}
    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}
    pointer allocate(size_type n, const void* = 0)
        {
            assert(data_ >= 0);
            if (time_to_throw >= throw_after) {
#ifndef _LIBCPP_NO_EXCEPTIONS
                throw std::bad_alloc();
#else
                std::terminate();
#endif
            }
            ++time_to_throw;
            ++alloc_count;
            return (pointer)::operator new (n * sizeof(T));
        }
    void deallocate(pointer p, size_type)
        {assert(data_ >= 0); --alloc_count; ::operator delete((void*)p); }
    size_type max_size() const throw()
        {return UINT_MAX / sizeof(T);}
#if TEST_STD_VER < 11
    void construct(pointer p, const T& val)
        {::new(static_cast<void*>(p)) T(val);}
#else
    template <class U> void construct(pointer p, U&& val)
        {::new(static_cast<void*>(p)) T(std::forward<U>(val));}
#endif
    void destroy(pointer p) {p->~T();}

    friend bool operator==(const non_default_test_allocator& x, const non_default_test_allocator& y)
        {return x.data_ == y.data_;}
    friend bool operator!=(const non_default_test_allocator& x, const non_default_test_allocator& y)
        {return !(x == y);}
};

template <>
class test_allocator<void>
    : public test_alloc_base
{
    int data_;

    template <class U> friend class test_allocator;
public:

    typedef unsigned                                                   size_type;
    typedef int                                                        difference_type;
    typedef void                                                       value_type;
    typedef value_type*                                                pointer;
    typedef const value_type*                                          const_pointer;

    template <class U> struct rebind {typedef test_allocator<U> other;};

    test_allocator() throw() : data_(0) {}
    explicit test_allocator(int i) throw() : data_(i) {}
    test_allocator(const test_allocator& a) throw()
        : data_(a.data_) {}
    template <class U> test_allocator(const test_allocator<U>& a) throw()
        : data_(a.data_) {}
    ~test_allocator() throw() {data_ = -1;}

    friend bool operator==(const test_allocator& x, const test_allocator& y)
        {return x.data_ == y.data_;}
    friend bool operator!=(const test_allocator& x, const test_allocator& y)
        {return !(x == y);}
};

template <class T>
class other_allocator
{
    int data_;

    template <class U> friend class other_allocator;

public:
    typedef T value_type;

    other_allocator() : data_(-1) {}
    explicit other_allocator(int i) : data_(i) {}
    template <class U> other_allocator(const other_allocator<U>& a)
        : data_(a.data_) {}
    T* allocate(std::size_t n)
        {return (T*)::operator new(n * sizeof(T));}
    void deallocate(T* p, std::size_t)
        {::operator delete((void*)p);}

    other_allocator select_on_container_copy_construction() const
        {return other_allocator(-2);}

    friend bool operator==(const other_allocator& x, const other_allocator& y)
        {return x.data_ == y.data_;}
    friend bool operator!=(const other_allocator& x, const other_allocator& y)
        {return !(x == y);}

    typedef std::true_type propagate_on_container_copy_assignment;
    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::true_type propagate_on_container_swap;

#ifdef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    std::size_t max_size() const
        {return UINT_MAX / sizeof(T);}
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE

};

#endif  // TEST_ALLOCATOR_H
