//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: with_system_cxx_lib=macosx
// REQUIRES: availability=macosx10.9 || availability=macosx10.10 || availability=macosx10.11 || availability=macosx10.12 || availability=macosx10.13 || availability=macosx10.14 || availability=macosx10.15

// Test the availability markup on the C++20 Synchronization Library
// additions to <atomic>.

#include <atomic>


int main(int, char**)
{
    {
        std::atomic<int> i(3);
        std::memory_order m = std::memory_order_relaxed;

        i.wait(4); // expected-error {{is unavailable}}
        i.wait(4, m); // expected-error {{is unavailable}}
        i.notify_one(); // expected-error {{is unavailable}}
        i.notify_all(); // expected-error {{is unavailable}}

        std::atomic_wait(&i, 4); // expected-error {{is unavailable}}
        std::atomic_wait_explicit(&i, 4, m); // expected-error {{is unavailable}}
        std::atomic_notify_one(&i); // expected-error {{is unavailable}}
        std::atomic_notify_all(&i); // expected-error {{is unavailable}}
    }

    {
        std::atomic<int> volatile i(3);
        std::memory_order m = std::memory_order_relaxed;

        i.wait(4); // expected-error {{is unavailable}}
        i.wait(4, m); // expected-error {{is unavailable}}
        i.notify_one(); // expected-error {{is unavailable}}
        i.notify_all(); // expected-error {{is unavailable}}

        std::atomic_wait(&i, 4); // expected-error {{is unavailable}}
        std::atomic_wait_explicit(&i, 4, m); // expected-error {{is unavailable}}
        std::atomic_notify_one(&i); // expected-error {{is unavailable}}
        std::atomic_notify_all(&i); // expected-error {{is unavailable}}
    }

    {
        std::atomic_flag flag;
        bool b = false;
        std::memory_order m = std::memory_order_relaxed;
        flag.wait(b); // expected-error {{is unavailable}}
        flag.wait(b, m); // expected-error {{is unavailable}}
        flag.notify_one(); // expected-error {{is unavailable}}
        flag.notify_all(); // expected-error {{is unavailable}}

        std::atomic_flag_wait(&flag, b); // expected-error {{is unavailable}}
        std::atomic_flag_wait_explicit(&flag, b, m); // expected-error {{is unavailable}}
        std::atomic_flag_notify_one(&flag); // expected-error {{is unavailable}}
        std::atomic_flag_notify_all(&flag); // expected-error {{is unavailable}}
    }

    {
        std::atomic_flag volatile flag;
        bool b = false;
        std::memory_order m = std::memory_order_relaxed;
        flag.wait(b); // expected-error {{is unavailable}}
        flag.wait(b, m); // expected-error {{is unavailable}}
        flag.notify_one(); // expected-error {{is unavailable}}
        flag.notify_all(); // expected-error {{is unavailable}}

        std::atomic_flag_wait(&flag, b); // expected-error {{is unavailable}}
        std::atomic_flag_wait_explicit(&flag, b, m); // expected-error {{is unavailable}}
        std::atomic_flag_notify_one(&flag); // expected-error {{is unavailable}}
        std::atomic_flag_notify_all(&flag); // expected-error {{is unavailable}}
    }
}
