// RUN: %clang_cc1 -verify -triple amdgcn -target-cpu verde -emit-llvm -o - %s
// RUN: %clang_cc1 -verify -triple amdgcn--opencl -target-cpu verde -emit-llvm -o - %s
// expected-no-diagnostics

// Record lowering was crashing on SI and newer targets, because it
// was using the wrong size for test::ptr.  Since global memory
// has 64-bit pointers, sizeof(test::ptr) should be 8.

struct test_as0 {int *ptr;};
constant int as0[sizeof(struct test_as0) == 4 ? 1 : -1] = { 0 };

struct test_as1 {global int *ptr;};
constant int as1[sizeof(struct test_as1) == 8 ? 1 : -1] = { 0 };

struct test_as2 {constant int *ptr;};
constant int as2[sizeof(struct test_as2) == 8 ? 1 : -1] = { 0 };

struct test_as3 {local int *ptr;};
constant int as3[sizeof(struct test_as3) == 4 ? 1 : -1] = { 0 };
