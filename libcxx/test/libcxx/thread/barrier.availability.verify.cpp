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

// Test the availability markup on std::barrier.

#include <barrier>
#include <utility>

struct CompletionF {
    void operator()() { }
};

int main(int, char**)
{
    // Availability markup on std::barrier<>
    {
        std::barrier<> b(10); // expected-error {{is unavailable}}
        auto token = b.arrive(); // expected-error {{is unavailable}}
        (void)b.arrive(10); // expected-error {{is unavailable}}
        b.wait(std::move(token)); // expected-error {{is unavailable}}
        b.arrive_and_wait(); // expected-error {{is unavailable}}
        b.arrive_and_drop(); // expected-error {{is unavailable}}
    }

    // Availability markup on std::barrier<CompletionF> with non-default CompletionF
    {
        std::barrier<CompletionF> b(10); // expected-error {{is unavailable}}
        auto token = b.arrive(); // expected-error {{is unavailable}}
        (void)b.arrive(10); // expected-error {{is unavailable}}
        b.wait(std::move(token)); // expected-error {{is unavailable}}
        b.arrive_and_wait(); // expected-error {{is unavailable}}
        b.arrive_and_drop(); // expected-error {{is unavailable}}
    }
}
