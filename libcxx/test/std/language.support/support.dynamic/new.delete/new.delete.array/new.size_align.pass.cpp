//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new[](std::size_t, std::align_val_t);

// UNSUPPORTED: c++03, c++11, c++14

// asan and msan will not call the new handler.
// UNSUPPORTED: sanitizer-new-delete

// We get availability markup errors when aligned allocation is missing
// XFAIL: availability-aligned_allocation-missing

// https://reviews.llvm.org/D129198 is not in AppleClang 14
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.13{{(.0)?}} && apple-clang-14

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <new>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <limits>

#include "test_macros.h"
#include "../types.h"

int new_handler_called = 0;

void my_new_handler() {
    ++new_handler_called;
    std::set_new_handler(nullptr);
}

int main(int, char**) {
    // Test that we can call the function directly
    {
        void* x = operator new[](10, static_cast<std::align_val_t>(64));
        assert(x != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(x) % 64 == 0);
        operator delete[](x, static_cast<std::align_val_t>(64));
    }

    // Test that the new handler is called if allocation fails
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        std::set_new_handler(my_new_handler);
        try {
            void* x = operator new[] (std::numeric_limits<std::size_t>::max(),
                                      static_cast<std::align_val_t>(32));
            (void)x;
            assert(false);
        } catch (std::bad_alloc const&) {
            assert(new_handler_called == 1);
        } catch (...) {
            assert(false);
        }
#endif
    }

    // Test that a new expression constructs the right object
    // and a delete expression deletes it
    {
        LifetimeInformation infos[3];
        TrackLifetimeOverAligned* x = new TrackLifetimeOverAligned[3]{infos[0], infos[1], infos[2]};
        assert(x != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(x) % alignof(TrackLifetimeOverAligned) == 0);

        void* addresses[3] = {&x[0], &x[1], &x[2]};
        assert(infos[0].address_constructed == addresses[0]);
        assert(infos[1].address_constructed == addresses[1]);
        assert(infos[2].address_constructed == addresses[2]);

        delete[] x;
        assert(infos[0].address_destroyed == addresses[0]);
        assert(infos[1].address_destroyed == addresses[1]);
        assert(infos[2].address_destroyed == addresses[2]);
    }

    return 0;
}
