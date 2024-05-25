//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: availability-synchronization_library-missing

// Test the availability markup on std::barrier.

#include <barrier>
#include <utility>

struct CompletionF {
    void operator()() { }
};

void f() {
    // Availability markup on std::barrier<>
    {
        std::barrier<> b(10);
        auto token = b.arrive(); // expected-warning {{'arrive' is only available on}}
        (void)b.arrive(10); // expected-warning {{'arrive' is only available on}}
        b.wait(std::move(token)); // expected-warning {{'wait' is only available on}}
        b.arrive_and_wait(); // expected-warning {{'arrive_and_wait' is only available on}}
        b.arrive_and_drop(); // expected-warning {{'arrive_and_drop' is only available on}}
    }

    // Availability markup on std::barrier<CompletionF> with non-default CompletionF
    {
        std::barrier<CompletionF> b(10);
        auto token = b.arrive(); // expected-warning {{'arrive' is only available on}}
        (void)b.arrive(10); // expected-warning {{'arrive' is only available on}}
        b.wait(std::move(token)); // expected-warning {{'wait' is only available on}}
        b.arrive_and_wait(); // expected-warning {{'arrive_and_wait' is only available on}}
        b.arrive_and_drop(); // expected-warning {{'arrive_and_drop' is only available on}}
    }
}
