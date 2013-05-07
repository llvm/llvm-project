#ifndef TEST_ALLOCATOR_H
#define TEST_ALLOCATOR_H

#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <new>
#include <climits>
#include <cassert>

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
            return (pointer)std::malloc(n * sizeof(T));
        }
    void deallocate(pointer p, size_type n)
        {assert(data_ >= 0); --alloc_count; std::free(p);}
    size_type max_size() const throw()
        {return UINT_MAX / sizeof(T);}
    void construct(pointer p, const T& val)
        {::new(p) T(val);}
    void destroy(pointer p) {p->~T();}

    friend bool operator==(const test_allocator& x, const test_allocator& y)
        {return x.data_ == y.data_;}
    friend bool operator!=(const test_allocator& x, const test_allocator& y)
        {return !(x == y);}
};

#endif  // TEST_ALLOCATOR_H
