// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

// A _BitInt whose padded storage integer has a larger alloc size than its
// store size (e.g. _BitInt(129) -> i192, alloc 32 != store 24) is laid out as
// a byte array by classic CodeGen.  That storage form is not yet implemented
// in CIR lowering, so it is reported instead of silently emitting a
// wrong-sized integer.

signed _BitInt(129) g129 = 1;

// CHECK: NYI: lowering global of a _BitInt with byte-array storage
