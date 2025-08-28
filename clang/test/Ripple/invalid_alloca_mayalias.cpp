// Fun fact: this compiles in optimization levels >= -O1 because the compiler
// figures that when promoting ThisBecomes2DAddr to a register it is replaced by
// the value of the alloca of ThisBecomes2D, thus allowing ThisBecomes2D to
// being promoted as well, removing the "MayAlias" ambiguity

// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: not %clang -S -g -O0 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clang -S -g -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-Other
// RUN: %clang -S -g -O2 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-Other
// RUN: %clang -S -g -O3 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-Other
// RUN: %clang -S -g -Os -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-Other


#include <ripple.h>

// CHECK-O0: error
// CHECK-Other: @fun
extern "C" void fun(size_t size, float A[size][size], float B[size][size]) {
    ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
    unsigned block_X = ripple_id(BS, 0);
    unsigned block_Y = ripple_id(BS, 1);
    unsigned size_X = ripple_get_block_size(BS, 0);
    unsigned size_Y = ripple_get_block_size(BS, 1);

    unsigned i;
    float ThisBecomes2D;
    float *const ThisBecomes2DAddr = &ThisBecomes2D;

    float Tmp = 0.f;

    for (i = 0; i < size; i += size_Y)
        Tmp += A[block_Y][i];

    // Here it may alias
    if (block_X < 4)
        *ThisBecomes2DAddr = Tmp;

    B[block_Y][block_X] = ThisBecomes2D;
}
