//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: with_system_cxx_lib=macosx10.9 || \
// REQUIRES: with_system_cxx_lib=macosx10.10 || \
// REQUIRES: with_system_cxx_lib=macosx10.11 || \
// REQUIRES: with_system_cxx_lib=macosx10.12 || \
// REQUIRES: with_system_cxx_lib=macosx10.13 || \
// REQUIRES: with_system_cxx_lib=macosx10.14 || \
// REQUIRES: with_system_cxx_lib=macosx10.15

// Test the availability markup on std::counting_semaphore and std::binary_semaphore.

#include <chrono>
#include <semaphore>


int main(int, char**)
{
    {
        // Tests for std::counting_semaphore with non-default template argument
        std::counting_semaphore<20> sem(10);
        sem.release(); // expected-error {{is unavailable}}
        sem.release(5); // expected-error {{is unavailable}}
        sem.acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-error 1-2 {{is unavailable}}
        sem.try_acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-error 1-2 {{is unavailable}}
    }
    {
        // Tests for std::counting_semaphore with default template argument
        std::counting_semaphore<> sem(10);
        sem.release(); // expected-error {{is unavailable}}
        sem.release(5); // expected-error {{is unavailable}}
        sem.acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-error 1-2 {{is unavailable}}
        sem.try_acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-error 1-2 {{is unavailable}}
    }
    {
        // Tests for std::binary_semaphore
        std::binary_semaphore sem(10);
        sem.release(); // expected-error {{is unavailable}}
        sem.release(5); // expected-error {{is unavailable}}
        sem.acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_for(std::chrono::milliseconds{3}); // expected-error 1-2 {{is unavailable}}
        sem.try_acquire(); // expected-error {{is unavailable}}
        sem.try_acquire_until(std::chrono::steady_clock::now()); // expected-error 1-2 {{is unavailable}}
    }
}
