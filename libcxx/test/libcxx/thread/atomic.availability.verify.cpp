//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: availability-synchronization_library-missing

// Test the availability markup on the C++20 Synchronization Library
// additions to <atomic>.

#include <atomic>

void f() {
    {
        std::atomic<int> i(3);
        std::memory_order m = std::memory_order_relaxed;

        i.wait(4); // expected-warning {{'wait' is only available on}}
        i.wait(4, m); // expected-warning {{'wait' is only available on}}
        i.notify_one(); // expected-warning {{'notify_one' is only available on}}
        i.notify_all(); // expected-warning {{'notify_all' is only available on}}

        std::atomic_wait(&i, 4); // expected-warning {{'atomic_wait<int>' is only available on}}
        std::atomic_wait_explicit(&i, 4, m); // expected-warning {{'atomic_wait_explicit<int>' is only available on}}
        std::atomic_notify_one(&i); // expected-warning {{'atomic_notify_one<int>' is only available on}}
        std::atomic_notify_all(&i); // expected-warning {{'atomic_notify_all<int>' is only available on}}
    }

    {
        std::atomic<int> volatile i(3);
        std::memory_order m = std::memory_order_relaxed;

        i.wait(4); // expected-warning {{'wait' is only available on}}
        i.wait(4, m); // expected-warning {{'wait' is only available on}}
        i.notify_one(); // expected-warning {{'notify_one' is only available on}}
        i.notify_all(); // expected-warning {{'notify_all' is only available on}}

        std::atomic_wait(&i, 4); // expected-warning {{'atomic_wait<int>' is only available on}}
        std::atomic_wait_explicit(&i, 4, m); // expected-warning {{'atomic_wait_explicit<int>' is only available on}}
        std::atomic_notify_one(&i); // expected-warning {{'atomic_notify_one<int>' is only available on}}
        std::atomic_notify_all(&i); // expected-warning {{'atomic_notify_all<int>' is only available on}}
    }

    {
        std::atomic_flag flag;
        bool b = false;
        std::memory_order m = std::memory_order_relaxed;

        flag.wait(b); // expected-warning {{'wait' is only available on}}
        flag.wait(b, m); // expected-warning {{'wait' is only available on}}
        flag.notify_one(); // expected-warning {{'notify_one' is only available on}}
        flag.notify_all(); // expected-warning {{'notify_all' is only available on}}

        std::atomic_flag_wait(&flag, b); // expected-warning {{'atomic_flag_wait' is only available on}}
        std::atomic_flag_wait_explicit(&flag, b, m); // expected-warning {{'atomic_flag_wait_explicit' is only available on}}
        std::atomic_flag_notify_one(&flag); // expected-warning {{'atomic_flag_notify_one' is only available on}}
        std::atomic_flag_notify_all(&flag); // expected-warning {{'atomic_flag_notify_all' is only available on}}
    }

    {
        std::atomic_flag volatile flag;
        bool b = false;
        std::memory_order m = std::memory_order_relaxed;

        flag.wait(b); // expected-warning {{'wait' is only available on}}
        flag.wait(b, m); // expected-warning {{'wait' is only available on}}
        flag.notify_one(); // expected-warning {{'notify_one' is only available on}}
        flag.notify_all(); // expected-warning {{'notify_all' is only available on}}

        std::atomic_flag_wait(&flag, b); // expected-warning {{'atomic_flag_wait' is only available on}}
        std::atomic_flag_wait_explicit(&flag, b, m); // expected-warning {{'atomic_flag_wait_explicit' is only available on}}
        std::atomic_flag_notify_one(&flag); // expected-warning {{'atomic_flag_notify_one' is only available on}}
        std::atomic_flag_notify_all(&flag); // expected-warning {{'atomic_flag_notify_all' is only available on}}
    }
}
