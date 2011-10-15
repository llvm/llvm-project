//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T desr, memory_order m = memory_order_seq_cst);
//     T load(memory_order m = memory_order_seq_cst) const volatile;
//     T load(memory_order m = memory_order_seq_cst) const;
//     operator T() const volatile;
//     operator T() const;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile;
//     T exchange(T desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f);
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst);
// 
//     atomic() = default;
//     constexpr atomic(T desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile;
//     T operator=(T);
// };

#include <atomic>
#include <cassert>

int main()
{
    {
        volatile std::atomic<bool> _;
        volatile std::atomic<bool> obj(true);
        assert(obj == true);
        std::atomic_init(&obj, false);
        assert(obj == false);
        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(obj.compare_exchange_weak(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == true);
        obj.store(true);
        assert(obj.compare_exchange_weak(x, false,
                                         std::memory_order_seq_cst,
                                         std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           std::memory_order_seq_cst,
                                           std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        std::atomic<bool> _;
        std::atomic<bool> obj(true);
        assert(obj == true);
        std::atomic_init(&obj, false);
        assert(obj == false);
        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(obj.compare_exchange_weak(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == true);
        obj.store(true);
        assert(obj.compare_exchange_weak(x, false,
                                         std::memory_order_seq_cst,
                                         std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           std::memory_order_seq_cst,
                                           std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
}
