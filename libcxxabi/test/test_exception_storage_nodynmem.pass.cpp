//===--------------- test_exception_storage_nodynmem.cpp ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cstdlib>

static bool OverwrittenCallocCalled = false;

// Override calloc to simulate exhaustion of dynamic memory
void *calloc(size_t, size_t) {
    OverwrittenCallocCalled = true;
    return 0;
}

int main(int argc, char *argv[]) {
    // Run the test a couple of times
    // to ensure that fallback memory doesn't leak.
    for (int I = 0; I < 1000; ++I)
        try {
            throw 42;
        } catch (...) {
        }

    assert(OverwrittenCallocCalled);
    return 0;
}
