//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIN_ALLOCATOR_H
#define MIN_ALLOCATOR_H

#include <cstddef>

#include "test_macros.h"

template <class T>
class bare_allocator
{
public:
    typedef T value_type;

    bare_allocator() TEST_NOEXCEPT {}

    template <class U>
    bare_allocator(bare_allocator<U>) TEST_NOEXCEPT {}

    T* allocate(std::size_t n)
    {
        return static_cast<T*>(::operator new(n*sizeof(T)));
    }

    void deallocate(T* p, std::size_t)
    {
        return ::operator delete(static_cast<void*>(p));
    }

    friend bool operator==(bare_allocator, bare_allocator) {return true;}
    friend bool operator!=(bare_allocator x, bare_allocator y) {return !(x == y);}
};


#if TEST_STD_VER >= 11

#include <memory>

template <class T> class min_pointer;
template <class T> class min_pointer<const T>;
template <> class min_pointer<void>;
template <> class min_pointer<const void>;
template <class T> class min_allocator;

template <>
class min_pointer<const void>
{
    const void* ptr_;
public:
    min_pointer() TEST_NOEXCEPT = default;
    min_pointer(std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
    template <class T>
    min_pointer(min_pointer<T> p) TEST_NOEXCEPT : ptr_(p.ptr_) {}

    explicit operator bool() const {return ptr_ != nullptr;}

    friend bool operator==(min_pointer x, min_pointer y) {return x.ptr_ == y.ptr_;}
    friend bool operator!=(min_pointer x, min_pointer y) {return !(x == y);}
    template <class U> friend class min_pointer;
};

template <>
class min_pointer<void>
{
    void* ptr_;
public:
    min_pointer() TEST_NOEXCEPT = default;
    min_pointer(std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
    template <class T,
              class = typename std::enable_if
                       <
                            !std::is_const<T>::value
                       >::type
             >
    min_pointer(min_pointer<T> p) TEST_NOEXCEPT : ptr_(p.ptr_) {}

    explicit operator bool() const {return ptr_ != nullptr;}

    friend bool operator==(min_pointer x, min_pointer y) {return x.ptr_ == y.ptr_;}
    friend bool operator!=(min_pointer x, min_pointer y) {return !(x == y);}
    template <class U> friend class min_pointer;
};

template <class T>
class min_pointer
{
    T* ptr_;

    explicit min_pointer(T* p) TEST_NOEXCEPT : ptr_(p) {}
public:
    min_pointer() TEST_NOEXCEPT = default;
    min_pointer(std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
    explicit min_pointer(min_pointer<void> p) TEST_NOEXCEPT : ptr_(static_cast<T*>(p.ptr_)) {}

    explicit operator bool() const {return ptr_ != nullptr;}

    typedef std::ptrdiff_t difference_type;
    typedef T& reference;
    typedef T* pointer;
    typedef T value_type;
    typedef std::random_access_iterator_tag iterator_category;

    reference operator*() const {return *ptr_;}
    pointer operator->() const {return ptr_;}

    min_pointer& operator++() {++ptr_; return *this;}
    min_pointer operator++(int) {min_pointer tmp(*this); ++ptr_; return tmp;}

    min_pointer& operator--() {--ptr_; return *this;}
    min_pointer operator--(int) {min_pointer tmp(*this); --ptr_; return tmp;}

    min_pointer& operator+=(difference_type n) {ptr_ += n; return *this;}
    min_pointer& operator-=(difference_type n) {ptr_ -= n; return *this;}

    min_pointer operator+(difference_type n) const
    {
        min_pointer tmp(*this);
        tmp += n;
        return tmp;
    }

    friend min_pointer operator+(difference_type n, min_pointer x)
    {
        return x + n;
    }

    min_pointer operator-(difference_type n) const
    {
        min_pointer tmp(*this);
        tmp -= n;
        return tmp;
    }

    friend difference_type operator-(min_pointer x, min_pointer y)
    {
        return x.ptr_ - y.ptr_;
    }

    reference operator[](difference_type n) const {return ptr_[n];}

    friend bool operator< (min_pointer x, min_pointer y) {return x.ptr_ < y.ptr_;}
    friend bool operator> (min_pointer x, min_pointer y) {return y < x;}
    friend bool operator<=(min_pointer x, min_pointer y) {return !(y < x);}
    friend bool operator>=(min_pointer x, min_pointer y) {return !(x < y);}

    static min_pointer pointer_to(T& t) {return min_pointer(std::addressof(t));}

    friend bool operator==(min_pointer x, min_pointer y) {return x.ptr_ == y.ptr_;}
    friend bool operator!=(min_pointer x, min_pointer y) {return !(x == y);}
    template <class U> friend class min_pointer;
    template <class U> friend class min_allocator;
};

template <class T>
class min_pointer<const T>
{
    const T* ptr_;

    explicit min_pointer(const T* p) : ptr_(p) {}
public:
    min_pointer() TEST_NOEXCEPT = default;
    min_pointer(std::nullptr_t) : ptr_(nullptr) {}
    min_pointer(min_pointer<T> p) : ptr_(p.ptr_) {}
    explicit min_pointer(min_pointer<const void> p) : ptr_(static_cast<const T*>(p.ptr_)) {}

    explicit operator bool() const {return ptr_ != nullptr;}

    typedef std::ptrdiff_t difference_type;
    typedef const T& reference;
    typedef const T* pointer;
    typedef const T value_type;
    typedef std::random_access_iterator_tag iterator_category;

    reference operator*() const {return *ptr_;}
    pointer operator->() const {return ptr_;}

    min_pointer& operator++() {++ptr_; return *this;}
    min_pointer operator++(int) {min_pointer tmp(*this); ++ptr_; return tmp;}

    min_pointer& operator--() {--ptr_; return *this;}
    min_pointer operator--(int) {min_pointer tmp(*this); --ptr_; return tmp;}

    min_pointer& operator+=(difference_type n) {ptr_ += n; return *this;}
    min_pointer& operator-=(difference_type n) {ptr_ -= n; return *this;}

    min_pointer operator+(difference_type n) const
    {
        min_pointer tmp(*this);
        tmp += n;
        return tmp;
    }

    friend min_pointer operator+(difference_type n, min_pointer x)
    {
        return x + n;
    }

    min_pointer operator-(difference_type n) const
    {
        min_pointer tmp(*this);
        tmp -= n;
        return tmp;
    }

    friend difference_type operator-(min_pointer x, min_pointer y)
    {
        return x.ptr_ - y.ptr_;
    }

    reference operator[](difference_type n) const {return ptr_[n];}

    friend bool operator< (min_pointer x, min_pointer y) {return x.ptr_ < y.ptr_;}
    friend bool operator> (min_pointer x, min_pointer y) {return y < x;}
    friend bool operator<=(min_pointer x, min_pointer y) {return !(y < x);}
    friend bool operator>=(min_pointer x, min_pointer y) {return !(x < y);}

    static min_pointer pointer_to(const T& t) {return min_pointer(std::addressof(t));}

    friend bool operator==(min_pointer x, min_pointer y) {return x.ptr_ == y.ptr_;}
    friend bool operator!=(min_pointer x, min_pointer y) {return !(x == y);}
    template <class U> friend class min_pointer;
};

template <class T>
inline
bool
operator==(min_pointer<T> x, std::nullptr_t)
{
    return !static_cast<bool>(x);
}

template <class T>
inline
bool
operator==(std::nullptr_t, min_pointer<T> x)
{
    return !static_cast<bool>(x);
}

template <class T>
inline
bool
operator!=(min_pointer<T> x, std::nullptr_t)
{
    return static_cast<bool>(x);
}

template <class T>
inline
bool
operator!=(std::nullptr_t, min_pointer<T> x)
{
    return static_cast<bool>(x);
}

template <class T>
class min_allocator
{
public:
    typedef T value_type;
    typedef min_pointer<T> pointer;

    min_allocator() = default;
    template <class U>
    min_allocator(min_allocator<U>) {}

    pointer allocate(std::ptrdiff_t n)
    {
        return pointer(static_cast<T*>(::operator new(n*sizeof(T))));
    }

    void deallocate(pointer p, std::ptrdiff_t)
    {
        return ::operator delete(p.ptr_);
    }

    friend bool operator==(min_allocator, min_allocator) {return true;}
    friend bool operator!=(min_allocator x, min_allocator y) {return !(x == y);}
};

#endif  // TEST_STD_VER >= 11

#endif  // MIN_ALLOCATOR_H
