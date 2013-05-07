#ifndef TEST_ALLOCATOR_H
#define TEST_ALLOCATOR_H

#include <cstddef>
#include <type_traits>
#include <cstdlib>
#include <new>
#include <climits>

class test_alloc_base
{
protected:
    static int count;
public:
    static int throw_after;
};

int test_alloc_base::count = 0;
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

    test_allocator() throw() : data_(-1) {}
    explicit test_allocator(int i) throw() : data_(i) {}
    test_allocator(const test_allocator& a) throw()
        : data_(a.data_) {}
    template <class U> test_allocator(const test_allocator<U>& a) throw()
        : data_(a.data_) {}
    ~test_allocator() throw() {data_ = 0;}
    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}
    pointer allocate(size_type n, const void* = 0)
        {
            if (count >= throw_after) {
#ifndef _LIBCPP_NO_EXCEPTIONS
                throw std::bad_alloc();
#else
                std::terminate();
#endif
            }
            ++count;
            return (pointer)std::malloc(n * sizeof(T));
        }
    void deallocate(pointer p, size_type n)
        {std::free(p);}
    size_type max_size() const throw()
        {return UINT_MAX / sizeof(T);}
    void construct(pointer p, const T& val)
        {::new(p) T(val);}
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    void construct(pointer p, T&& val)
        {::new(p) T(std::move(val));}
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    void destroy(pointer p) {p->~T();}

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
        {return (T*)std::malloc(n * sizeof(T));}
    void deallocate(T* p, std::size_t n)
        {std::free(p);}

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
