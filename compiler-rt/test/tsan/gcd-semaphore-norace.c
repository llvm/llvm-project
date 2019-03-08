// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: dispatch

#include <dispatch/dispatch.h>

#include <stdio.h>

long global;

int main() {
    fprintf(stderr, "Hello world.");

    global = 42;

    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{

        global = 43;
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    global = 44;

    fprintf(stderr, "Done.");
    return 0;
}

// CHECK: Hello world.
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
