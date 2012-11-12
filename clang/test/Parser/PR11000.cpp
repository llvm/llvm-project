// RUN: %clang_cc1 -std=c++11 %s 2>&1 | FileCheck %s

// PR11000: Don't crash.
class tuple<>
{
    template <class _Alloc>
        tuple(allocator_arg_t, const _Alloc&) {}

// CHECK: 6 errors generated.
