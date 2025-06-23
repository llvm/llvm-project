//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new(std::size_t);

// asan and msan will not call the new handler.
// UNSUPPORTED: sanitizer-new-delete

// GCC warns about allocating numeric_limits<size_t>::max() being too large (which we test here)
// ADDITIONAL_COMPILE_FLAGS(gcc): -Wno-alloc-size-larger-than

#include <new>
#include <cstddef>
#include <cassert>
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
        void* x = operator new(10);
        assert(x != nullptr);
        operator delete(x);
    }

    // Test that the new handler is called if allocation fails
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        std::set_new_handler(my_new_handler);
        try {
            void* x = operator new(std::numeric_limits<std::size_t>::max());
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
        LifetimeInformation info;
        TrackLifetime* x = new TrackLifetime(info);
        assert(x != nullptr);
        assert(info.address_constructed == x);

        delete x;
        assert(info.address_destroyed == x);
    }

    return 0;
}
