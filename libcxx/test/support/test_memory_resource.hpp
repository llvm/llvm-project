//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MEMORY_RESOURCE_HPP
#define SUPPORT_TEST_MEMORY_RESOURCE_HPP

#include <experimental/memory_resource>
#include <memory>
#include <type_traits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include "test_macros.h"

struct AllocController;
    // 'AllocController' is a concrete type that instruments and controls the
    // behavior of of test allocators.

template <class T>
class CountingAllocator;
    // 'CountingAllocator' is an basic implementation of the 'Allocator'
    // requirements that use the 'AllocController' interface.

template <class T>
class MinAlignAllocator;
    // 'MinAlignAllocator' is an instrumented test type which implements the
    // 'Allocator' requirements. 'MinAlignAllocator' ensures that it *never*
    // returns a pointer to over-aligned storage. For example
    // 'MinAlignPointer<char>{}.allocate(...)' will never a 2-byte aligned
    // pointer.

template <class T>
class NullAllocator;
    // 'NullAllocator' is an instrumented test type which implements the
    // 'Allocator' requirements except that 'allocator' and 'deallocate' are
    // nops.


#define DISALLOW_COPY(Type) \
  Type(Type const&) = delete; \
  Type& operator=(Type const&) = delete

constexpr std::size_t MaxAlignV = alignof(std::max_align_t);

struct TestException {};

struct AllocController {
    int copy_constructed = 0;
    int move_constructed = 0;

    int alive = 0;
    int alloc_count = 0;
    int dealloc_count = 0;
    int is_equal_count = 0;

    std::size_t alive_size;
    std::size_t allocated_size;
    std::size_t deallocated_size;

    std::size_t last_size = 0;
    std::size_t last_align = 0;
    void * last_pointer = 0;

    std::size_t last_alloc_size = 0;
    std::size_t last_alloc_align = 0;
    void * last_alloc_pointer = nullptr;

    std::size_t last_dealloc_size = 0;
    std::size_t last_dealloc_align = 0;
    void * last_dealloc_pointer = nullptr;

    bool throw_on_alloc = false;

    AllocController() = default;

    void countAlloc(void* p, size_t s, size_t a) {
        ++alive;
        ++alloc_count;
        alive_size += s;
        allocated_size += s;
        last_pointer = last_alloc_pointer = p;
        last_size = last_alloc_size = s;
        last_align = last_alloc_align = a;
    }

    void countDealloc(void* p, size_t s, size_t a) {
        --alive;
        ++dealloc_count;
        alive_size -= s;
        deallocated_size += s;
        last_pointer = last_dealloc_pointer = p;
        last_size = last_dealloc_size = s;
        last_align = last_dealloc_align = a;
    }

    void reset() { std::memset(this, 0, sizeof(*this)); }

public:
    bool checkAlloc(void* p, size_t s, size_t a) const {
        return p == last_alloc_pointer &&
               s == last_alloc_size &&
               a == last_alloc_align;
    }

    bool checkAlloc(void* p, size_t s) const {
        return p == last_alloc_pointer &&
               s == last_alloc_size;
    }

    bool checkAllocAtLeast(void* p, size_t s, size_t a) const {
        return p == last_alloc_pointer &&
               s <= last_alloc_size &&
               a <= last_alloc_align;
    }

    bool checkAllocAtLeast(void* p, size_t s) const {
        return p == last_alloc_pointer &&
               s <= last_alloc_size;
    }

    bool checkDealloc(void* p, size_t s, size_t a) const {
        return p == last_dealloc_pointer &&
               s == last_dealloc_size &&
               a == last_dealloc_align;
    }

    bool checkDealloc(void* p, size_t s) const {
        return p == last_dealloc_pointer &&
               s == last_dealloc_size;
    }

    bool checkDeallocMatchesAlloc() const {
        return last_dealloc_pointer == last_alloc_pointer &&
               last_dealloc_size == last_alloc_size &&
               last_dealloc_align == last_alloc_align;
    }

    void countIsEqual() {
        ++is_equal_count;
    }

    bool checkIsEqualCalledEq(int n) const {
        return is_equal_count == n;
    }
private:
  DISALLOW_COPY(AllocController);
};

template <class T>
class CountingAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    CountingAllocator() = delete;
    explicit CountingAllocator(AllocController& PP) : P(&PP) {}

    CountingAllocator(CountingAllocator const& other) : P(other.P) {
        P->copy_constructed += 1;
    }

    CountingAllocator(CountingAllocator&& other) : P(other.P) {
        P->move_constructed += 1;
    }

    template <class U>
    CountingAllocator(CountingAllocator<U> const& other) TEST_NOEXCEPT : P(other.P) {
        P->copy_constructed += 1;
    }

    template <class U>
    CountingAllocator(CountingAllocator<U>&& other) TEST_NOEXCEPT : P(other.P) {
        P->move_constructed += 1;
    }

    T* allocate(std::size_t n)
    {
        void* ret = ::operator new(n*sizeof(T));
        P->countAlloc(ret, n*sizeof(T), alignof(T));
        return static_cast<T*>(ret);
    }

    void deallocate(T* p, std::size_t n)
    {
        void* vp = static_cast<void*>(p);
        P->countDealloc(vp, n*sizeof(T), alignof(T));
        ::operator delete(vp);
    }

    AllocController& getController() const { return *P; }

private:
    template <class Tp> friend class CountingAllocator;
    AllocController *P;
};

template <class T, class U>
inline bool operator==(CountingAllocator<T> const& x,
                       CountingAllocator<U> const& y) {
    return &x.getController() == &y.getController();
}

template <class T, class U>
inline bool operator!=(CountingAllocator<T> const& x,
                       CountingAllocator<U> const& y) {
    return !(x == y);
}

template <class T>
class MinAlignedAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;

    MinAlignedAllocator() = delete;

    explicit MinAlignedAllocator(AllocController& R) : P(&R) {}

    MinAlignedAllocator(MinAlignedAllocator const& other) : P(other.P) {
        P->copy_constructed += 1;
    }

    MinAlignedAllocator(MinAlignedAllocator&& other) : P(other.P) {
        P->move_constructed += 1;
    }

    template <class U>
    MinAlignedAllocator(MinAlignedAllocator<U> const& other) TEST_NOEXCEPT : P(other.P) {
        P->copy_constructed += 1;
    }

    template <class U>
    MinAlignedAllocator(MinAlignedAllocator<U>&& other) TEST_NOEXCEPT : P(other.P) {
        P->move_constructed += 1;
    }

    T* allocate(std::size_t n) {
        char* aligned_ptr = (char*)::operator new(alloc_size(n*sizeof(T)));
        assert(is_max_aligned(aligned_ptr));

        char* unaligned_ptr = aligned_ptr + alignof(T);
        assert(is_min_aligned(unaligned_ptr));

        P->countAlloc(unaligned_ptr, n * sizeof(T), alignof(T));

        return ((T*)unaligned_ptr);
    }

    void deallocate(T* p, std::size_t n) {
        assert(is_min_aligned(p));

        char* aligned_ptr = ((char*)p) - alignof(T);
        assert(is_max_aligned(aligned_ptr));

        P->countDealloc(p, n*sizeof(T), alignof(T));

        return ::operator delete(static_cast<void*>(aligned_ptr));
    }

    AllocController& getController() const { return *P; }

private:
    static const std::size_t BlockSize = alignof(std::max_align_t);

    static std::size_t alloc_size(std::size_t s) {
        std::size_t bytes = (s + BlockSize - 1) & ~(BlockSize - 1);
        bytes += BlockSize;
        assert(bytes % BlockSize == 0);
        return bytes;
    }

    static bool is_max_aligned(void* p) {
        return reinterpret_cast<std::uintptr_t>(p) % BlockSize == 0;
    }

    static bool is_min_aligned(void* p) {
        if (alignof(T) == BlockSize) {
            return is_max_aligned(p);
        } else {
            return reinterpret_cast<std::uintptr_t>(p) % BlockSize == alignof(T);
        }
    }

    template <class Tp> friend class MinAlignedAllocator;
    mutable AllocController *P;
};


template <class T, class U>
inline bool operator==(MinAlignedAllocator<T> const& x,
                       MinAlignedAllocator<U> const& y) {
    return &x.getController() == &y.getController();
}

template <class T, class U>
inline bool operator!=(MinAlignedAllocator<T> const& x,
                       MinAlignedAllocator<U> const& y) {
    return !(x == y);
}

template <class T>
class NullAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    NullAllocator() = delete;
    explicit NullAllocator(AllocController& PP) : P(&PP) {}

    NullAllocator(NullAllocator const& other) : P(other.P) {
        P->copy_constructed += 1;
    }

    NullAllocator(NullAllocator&& other) : P(other.P) {
        P->move_constructed += 1;
    }

    template <class U>
    NullAllocator(NullAllocator<U> const& other) TEST_NOEXCEPT : P(other.P) {
        P->copy_constructed += 1;
    }

    template <class U>
    NullAllocator(NullAllocator<U>&& other) TEST_NOEXCEPT : P(other.P) {
        P->move_constructed += 1;
    }

    T* allocate(std::size_t n)
    {
        P->countAlloc(nullptr, n*sizeof(T), alignof(T));
        return nullptr;
    }

    void deallocate(T* p, std::size_t n)
    {
        void* vp = static_cast<void*>(p);
        P->countDealloc(vp, n*sizeof(T), alignof(T));
    }

    AllocController& getController() const { return *P; }

private:
    template <class Tp> friend class NullAllocator;
    AllocController *P;
};

template <class T, class U>
inline bool operator==(NullAllocator<T> const& x,
                       NullAllocator<U> const& y) {
    return &x.getController() == &y.getController();
}

template <class T, class U>
inline bool operator!=(NullAllocator<T> const& x,
                       NullAllocator<U> const& y) {
    return !(x == y);
}



template <class ProviderT, int = 0>
class TestResourceImp : public std::experimental::pmr::memory_resource
{
public:
    static int resource_alive;
    static int resource_constructed;
    static int resource_destructed;

    static void resetStatics() {
        assert(resource_alive == 0);
        resource_alive = 0;
        resource_constructed = 0;
        resource_destructed = 0;
    }

    using memory_resource = std::experimental::pmr::memory_resource;
    using Provider = ProviderT;

    int value;

    explicit TestResourceImp(int val = 0) : value(val) {
        ++resource_alive;
        ++resource_constructed;
    }

    ~TestResourceImp() noexcept {
        --resource_alive;
        ++resource_destructed;
    }

    void reset() { C.reset(); P.reset(); }
    AllocController& getController() { return C; }

    bool checkAlloc(void* p, std::size_t s, std::size_t a) const
      { return C.checkAlloc(p, s, a); }

    bool checkDealloc(void* p, std::size_t s, std::size_t a) const
      { return C.checkDealloc(p, s, a); }

    bool checkIsEqualCalledEq(int n) const { return C.checkIsEqualCalledEq(n); }

protected:
    virtual void * do_allocate(std::size_t s, std::size_t a) {
        if (C.throw_on_alloc) {
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw TestException{};
#else
            assert(false);
#endif
        }
        void* ret = P.allocate(s, a);
        C.countAlloc(ret, s, a);
        return ret;
    }

    virtual void do_deallocate(void * p, std::size_t s, std::size_t a) {
        C.countDealloc(p, s, a);
        P.deallocate(p, s, a);
    }

    virtual bool do_is_equal(memory_resource const & other) const noexcept {
        C.countIsEqual();
        TestResourceImp const * o = dynamic_cast<TestResourceImp const *>(&other);
        return o && o->value == value;
    }
private:
    mutable AllocController C;
    mutable Provider P;
    DISALLOW_COPY(TestResourceImp);
};

template <class Provider, int N>
int TestResourceImp<Provider, N>::resource_alive = 0;

template <class Provider, int N>
int TestResourceImp<Provider, N>::resource_constructed = 0;

template <class Provider, int N>
int TestResourceImp<Provider, N>::resource_destructed = 0;


struct NullProvider {
    NullProvider() {}
    void* allocate(size_t, size_t) { return nullptr; }
    void deallocate(void*, size_t, size_t) {}
    void reset() {}
private:
    DISALLOW_COPY(NullProvider);
};

struct NewDeleteProvider {
    NewDeleteProvider() {}
    void* allocate(size_t s, size_t) { return ::operator new(s); }
    void deallocate(void* p, size_t, size_t) { ::operator delete(p); }
    void reset() {}
private:
    DISALLOW_COPY(NewDeleteProvider);
};

template <size_t Size = 4096 * 10> // 10 pages worth of memory.
struct BufferProvider {
    char buffer[Size];
    void* next = &buffer;
    size_t space = Size;

    BufferProvider() {}

    void* allocate(size_t s, size_t a) {
        void* ret = std::align(s, a, next, space);
        if (ret == nullptr) {
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw std::bad_alloc();
#else
            assert(false);
#endif
        }

        return ret;
    }

    void deallocate(void*, size_t, size_t) {}

    void reset() {
        next = &buffer;
        space = Size;
    }
private:
    DISALLOW_COPY(BufferProvider);
};

using NullResource = TestResourceImp<NullProvider, 0>;
using NewDeleteResource = TestResourceImp<NewDeleteProvider, 0>;
using TestResource  = TestResourceImp<BufferProvider<>, 0>;
using TestResource1 = TestResourceImp<BufferProvider<>, 1>;
using TestResource2 = TestResourceImp<BufferProvider<>, 2>;


#endif /* SUPPORT_TEST_MEMORY_RESOURCE_HPP */
