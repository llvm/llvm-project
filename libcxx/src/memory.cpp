//===------------------------ memory.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_BUILDING_MEMORY
#include "memory"
#include "mutex"
#include "thread"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace
{

template <class T>
inline T
increment(T& t) _NOEXCEPT
{
    return __sync_add_and_fetch(&t, 1);
}

template <class T>
inline T
decrement(T& t) _NOEXCEPT
{
    return __sync_add_and_fetch(&t, -1);
}

}  // namespace

const allocator_arg_t allocator_arg = allocator_arg_t();

bad_weak_ptr::~bad_weak_ptr() _NOEXCEPT {}

const char*
bad_weak_ptr::what() const _NOEXCEPT
{
    return "bad_weak_ptr";
}

__shared_count::~__shared_count()
{
}

void
__shared_count::__add_shared() _NOEXCEPT
{
    increment(__shared_owners_);
}

bool
__shared_count::__release_shared() _NOEXCEPT
{
    if (decrement(__shared_owners_) == -1)
    {
        __on_zero_shared();
        return true;
    }
    return false;
}

__shared_weak_count::~__shared_weak_count()
{
}

void
__shared_weak_count::__add_shared() _NOEXCEPT
{
    __shared_count::__add_shared();
}

void
__shared_weak_count::__add_weak() _NOEXCEPT
{
    increment(__shared_weak_owners_);
}

void
__shared_weak_count::__release_shared() _NOEXCEPT
{
    if (__shared_count::__release_shared())
        __release_weak();
}

void
__shared_weak_count::__release_weak() _NOEXCEPT
{
    if (decrement(__shared_weak_owners_) == -1)
        __on_zero_shared_weak();
}

__shared_weak_count*
__shared_weak_count::lock() _NOEXCEPT
{
    long object_owners = __shared_owners_;
    while (object_owners != -1)
    {
        if (__sync_bool_compare_and_swap(&__shared_owners_,
                                         object_owners,
                                         object_owners+1))
            return this;
        object_owners = __shared_owners_;
    }
    return 0;
}

#ifndef _LIBCPP_NO_RTTI

const void*
__shared_weak_count::__get_deleter(const type_info&) const _NOEXCEPT
{
    return 0;
}

#endif  // _LIBCPP_NO_RTTI

#if __has_feature(cxx_atomic)

static const std::size_t __sp_mut_count = 16;
static pthread_mutex_t mut_back_imp[__sp_mut_count] =
{
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER
};

static mutex* mut_back = reinterpret_cast<std::mutex*>(mut_back_imp);

_LIBCPP_CONSTEXPR __sp_mut::__sp_mut(void* p) _NOEXCEPT
   : __lx(p)
{
}

void
__sp_mut::lock() _NOEXCEPT
{
    mutex& m = *static_cast<mutex*>(__lx);
    unsigned count = 0;
    while (!m.try_lock())
    {
        if (++count > 16)
        {
            m.lock();
            break;
        }
        this_thread::yield();
    }
}

void
__sp_mut::unlock() _NOEXCEPT
{
    static_cast<mutex*>(__lx)->unlock();
}

__sp_mut&
__get_sp_mut(const void* p)
{
    static __sp_mut muts[__sp_mut_count] 
    {
        &mut_back[ 0], &mut_back[ 1], &mut_back[ 2], &mut_back[ 3],
        &mut_back[ 4], &mut_back[ 5], &mut_back[ 6], &mut_back[ 7],
        &mut_back[ 8], &mut_back[ 9], &mut_back[10], &mut_back[11],
        &mut_back[12], &mut_back[13], &mut_back[14], &mut_back[15]
    };
    return muts[hash<const void*>()(p) & (__sp_mut_count-1)];
}

#endif // __has_feature(cxx_atomic)

void
declare_reachable(void*)
{
}

void
declare_no_pointers(char*, size_t)
{
}

void
undeclare_no_pointers(char*, size_t)
{
}

pointer_safety
get_pointer_safety() _NOEXCEPT
{
    return pointer_safety::relaxed;
}

void*
__undeclare_reachable(void* p)
{
    return p;
}

void*
align(size_t alignment, size_t size, void*& ptr, size_t& space)
{
    void* r = nullptr;
    if (size <= space)
    {
        char* p1 = static_cast<char*>(ptr);
        char* p2 = (char*)((size_t)(p1 + (alignment - 1)) & -alignment);
        size_t d = static_cast<size_t>(p2 - p1);
        if (d <= space - size)
        {
            r = p2;
            ptr = r;
            space -= d;
        }
    }
    return r;
}

_LIBCPP_END_NAMESPACE_STD
