//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// REQUIRES: availability-synchronization_library-missing

// Test the availability markup on std::latch.

#include <latch>

void f() {
    std::latch latch(10);
    latch.count_down(); // expected-warning {{'count_down' is only available on}}
    latch.count_down(3); // expected-warning {{'count_down' is only available on}}
    latch.wait(); // expected-warning {{'wait' is only available on}}
    latch.arrive_and_wait(); // expected-warning {{'arrive_and_wait' is only available on}}
    latch.arrive_and_wait(3); // expected-warning {{'arrive_and_wait' is only available on}}
}
