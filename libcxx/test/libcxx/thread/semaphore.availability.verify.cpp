//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: availability-synchronization_library-missing

// Test the availability markup on std::counting_semaphore and std::binary_semaphore.

#include <chrono>
#include <semaphore>

void f() {
    {
        // Tests for std::counting_semaphore with non-default template argument
        std::counting_semaphore<20> sem(10);
        sem.release(); // expected-warning {{'release' is only available on}}
        sem.release(5); // expected-warning {{'release' is only available on}}
        sem.acquire(); // expected-warning {{'acquire' is only available on}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-warning-re {{'try_acquire_for<{{.+}}>' is only available on}}
        sem.try_acquire(); // expected-warning {{'try_acquire' is only available on}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-warning-re {{'try_acquire_until<{{.+}}>' is only available on}}
    }
    {
        // Tests for std::counting_semaphore with default template argument
        std::counting_semaphore<> sem(10);
        sem.release(); // expected-warning {{'release' is only available on}}
        sem.release(5); // expected-warning {{'release' is only available on}}
        sem.acquire(); // expected-warning {{'acquire' is only available on}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-warning-re {{'try_acquire_for<{{.+}}>' is only available on}}
        sem.try_acquire(); // expected-warning {{'try_acquire' is only available on}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-warning-re {{'try_acquire_until<{{.+}}>' is only available on}}
    }
    {
        // Tests for std::binary_semaphore
        std::binary_semaphore sem(10);
        sem.release(); // expected-warning {{'release' is only available on}}
        sem.release(5); // expected-warning {{'release' is only available on}}
        sem.acquire(); // expected-warning {{'acquire' is only available on}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-warning-re {{'try_acquire_for<{{.+}}>' is only available on}}
        sem.try_acquire(); // expected-warning {{'try_acquire' is only available on}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-warning-re {{'try_acquire_until<{{.+}}>' is only available on}}
    }
}
