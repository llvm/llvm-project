// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-reduced-module-interface %t/lock.cppm -o %t/lock.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-reduced-module-interface %t/queue.cppm -o %t/queue.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-reduced-module-interface %t/threadpool.cppm -o %t/threadpool.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/user.cppm -o %t/user.pcm -fprebuilt-module-path=%t -emit-llvm -o - | FileCheck %t/user.cppm

//--- lock.h
#pragma once

namespace std {

struct mutex {
    void lock() {}
    void unlock() {}
};

template <class T>
struct lock_guard {
    lock_guard(T &m) : m(m) {
        m.lock();
    }
    ~lock_guard() {
        m.unlock();
    }

    T &m;
};
}

//--- lock.cppm
module;
#include "lock.h"
export module lock;
namespace std {
    export using ::std::mutex;
    export using ::std::lock_guard;
}

//--- queue.cppm
export module queue;
import lock;

export template <class T>
class queue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(m_mutex);
    }
    T pop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return T{};
    }

private:
    std::mutex m_mutex;
};

//--- threadpool.cppm
export module threadpool;
import queue;

export class threadpool {
public:

    void add_task();
    int pop_task();
private:
    queue<int> m_queue;
};

inline void threadpool::add_task() {
    m_queue.push(42);
}

inline int threadpool::pop_task() {
    return m_queue.pop();
}

//--- user.cppm
export module user;
import threadpool;

export class user {
public:
    void run() {
        m_threadpool.add_task();
        m_threadpool.pop_task();
    }
private:
    threadpool m_threadpool;
};

// CHECK: define {{.*}}@_ZNSt10lock_guardISt5mutexED1Ev(
