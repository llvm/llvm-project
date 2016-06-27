//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef COUNT_NEW_HPP
#define COUNT_NEW_HPP

# include <cstdlib>
# include <cassert>
# include <new>

#include "test_macros.h"

#if defined(TEST_HAS_SANITIZERS)
#define DISABLE_NEW_COUNT
#endif

namespace detail
{
   TEST_NORETURN
   inline void throw_bad_alloc_helper() {
#ifndef TEST_HAS_NO_EXCEPTIONS
       throw std::bad_alloc();
#else
       std::abort();
#endif
   }
}

class MemCounter
{
public:
    // Make MemCounter super hard to accidentally construct or copy.
    class MemCounterCtorArg_ {};
    explicit MemCounter(MemCounterCtorArg_) { reset(); }

private:
    MemCounter(MemCounter const &);
    MemCounter & operator=(MemCounter const &);

public:
    // All checks return true when disable_checking is enabled.
    static const bool disable_checking;

    // Disallow any allocations from occurring. Useful for testing that
    // code doesn't perform any allocations.
    bool disable_allocations;

    // number of allocations to throw after. Default (unsigned)-1. If
    // throw_after has the default value it will never be decremented.
    static const unsigned never_throw_value = static_cast<unsigned>(-1);
    unsigned throw_after;

    int outstanding_new;
    int new_called;
    int delete_called;
    int last_new_size;

    int outstanding_array_new;
    int new_array_called;
    int delete_array_called;
    int last_new_array_size;

public:
    void newCalled(std::size_t s)
    {
        assert(disable_allocations == false);
        assert(s);
        if (throw_after == 0) {
            throw_after = never_throw_value;
            detail::throw_bad_alloc_helper();
        } else if (throw_after != never_throw_value) {
            --throw_after;
        }
        ++new_called;
        ++outstanding_new;
        last_new_size = s;
    }

    void deleteCalled(void * p)
    {
        assert(p);
        --outstanding_new;
        ++delete_called;
    }

    void newArrayCalled(std::size_t s)
    {
        assert(disable_allocations == false);
        assert(s);
        if (throw_after == 0) {
            throw_after = never_throw_value;
            detail::throw_bad_alloc_helper();
        } else {
            // don't decrement throw_after here. newCalled will end up doing that.
        }
        ++outstanding_array_new;
        ++new_array_called;
        last_new_array_size = s;
    }

    void deleteArrayCalled(void * p)
    {
        assert(p);
        --outstanding_array_new;
        ++delete_array_called;
    }

    void disableAllocations()
    {
        disable_allocations = true;
    }

    void enableAllocations()
    {
        disable_allocations = false;
    }


    void reset()
    {
        disable_allocations = false;
        throw_after = never_throw_value;

        outstanding_new = 0;
        new_called = 0;
        delete_called = 0;
        last_new_size = 0;

        outstanding_array_new = 0;
        new_array_called = 0;
        delete_array_called = 0;
        last_new_array_size = 0;
    }

public:
    bool checkOutstandingNewEq(int n) const
    {
        return disable_checking || n == outstanding_new;
    }

    bool checkOutstandingNewNotEq(int n) const
    {
        return disable_checking || n != outstanding_new;
    }

    bool checkNewCalledEq(int n) const
    {
        return disable_checking || n == new_called;
    }

    bool checkNewCalledNotEq(int n) const
    {
        return disable_checking || n != new_called;
    }

    bool checkNewCalledGreaterThan(int n) const
    {
        return disable_checking || new_called > n;
    }

    bool checkDeleteCalledEq(int n) const
    {
        return disable_checking || n == delete_called;
    }

    bool checkDeleteCalledNotEq(int n) const
    {
        return disable_checking || n != delete_called;
    }

    bool checkLastNewSizeEq(int n) const
    {
        return disable_checking || n == last_new_size;
    }

    bool checkLastNewSizeNotEq(int n) const
    {
        return disable_checking || n != last_new_size;
    }

    bool checkOutstandingArrayNewEq(int n) const
    {
        return disable_checking || n == outstanding_array_new;
    }

    bool checkOutstandingArrayNewNotEq(int n) const
    {
        return disable_checking || n != outstanding_array_new;
    }

    bool checkNewArrayCalledEq(int n) const
    {
        return disable_checking || n == new_array_called;
    }

    bool checkNewArrayCalledNotEq(int n) const
    {
        return disable_checking || n != new_array_called;
    }

    bool checkDeleteArrayCalledEq(int n) const
    {
        return disable_checking || n == delete_array_called;
    }

    bool checkDeleteArrayCalledNotEq(int n) const
    {
        return disable_checking || n != delete_array_called;
    }

    bool checkLastNewArraySizeEq(int n) const
    {
        return disable_checking || n == last_new_array_size;
    }

    bool checkLastNewArraySizeNotEq(int n) const
    {
        return disable_checking || n != last_new_array_size;
    }
};

#ifdef DISABLE_NEW_COUNT
  const bool MemCounter::disable_checking = true;
#else
  const bool MemCounter::disable_checking = false;
#endif

MemCounter globalMemCounter((MemCounter::MemCounterCtorArg_()));

#ifndef DISABLE_NEW_COUNT
void* operator new(std::size_t s) throw(std::bad_alloc)
{
    globalMemCounter.newCalled(s);
    void* ret = std::malloc(s);
    if (ret == nullptr)
        detail::throw_bad_alloc_helper();
    return ret;
}

void  operator delete(void* p) throw()
{
    globalMemCounter.deleteCalled(p);
    std::free(p);
}


void* operator new[](std::size_t s) throw(std::bad_alloc)
{
    globalMemCounter.newArrayCalled(s);
    return operator new(s);
}


void operator delete[](void* p) throw()
{
    globalMemCounter.deleteArrayCalled(p);
    operator delete(p);
}

#endif // DISABLE_NEW_COUNT


struct DisableAllocationGuard {
    explicit DisableAllocationGuard(bool disable = true) : m_disabled(disable)
    {
        // Don't re-disable if already disabled.
        if (globalMemCounter.disable_allocations == true) m_disabled = false;
        if (m_disabled) globalMemCounter.disableAllocations();
    }

    void release() {
        if (m_disabled) globalMemCounter.enableAllocations();
        m_disabled = false;
    }

    ~DisableAllocationGuard() {
        release();
    }

private:
    bool m_disabled;

    DisableAllocationGuard(DisableAllocationGuard const&);
    DisableAllocationGuard& operator=(DisableAllocationGuard const&);
};


struct RequireAllocationGuard {
    explicit RequireAllocationGuard(std::size_t RequireAtLeast = 1)
            : m_req_alloc(RequireAtLeast),
              m_new_count_on_init(globalMemCounter.new_called),
              m_outstanding_new_on_init(globalMemCounter.outstanding_new),
              m_exactly(false)
    {
    }

    void requireAtLeast(std::size_t N) { m_req_alloc = N; m_exactly = false; }
    void requireExactly(std::size_t N) { m_req_alloc = N; m_exactly = true; }

    ~RequireAllocationGuard() {
        assert(globalMemCounter.checkOutstandingNewEq(m_outstanding_new_on_init));
        std::size_t Expect = m_new_count_on_init + m_req_alloc;
        assert(globalMemCounter.checkNewCalledEq(Expect) ||
               (!m_exactly && globalMemCounter.checkNewCalledGreaterThan(Expect)));
    }

private:
    std::size_t m_req_alloc;
    const std::size_t m_new_count_on_init;
    const std::size_t m_outstanding_new_on_init;
    bool m_exactly;
    RequireAllocationGuard(RequireAllocationGuard const&);
    RequireAllocationGuard& operator=(RequireAllocationGuard const&);
};

#endif /* COUNT_NEW_HPP */
