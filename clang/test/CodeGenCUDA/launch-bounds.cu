// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s

#include "Inputs/cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2

// Test both max threads per block and Min cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel1()
{
}
}

// CHECK: !{{[0-9]+}} = !{void ()* @Kernel1, !"maxntidx", i32 256}
// CHECK: !{{[0-9]+}} = !{void ()* @Kernel1, !"minctasm", i32 2}

// Test only max threads per block. Min cta per sm defaults to 0, and
// CodeGen doesn't output a zero value for minctasm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK )
Kernel2()
{
}
}

// CHECK: !{{[0-9]+}} = !{void ()* @Kernel2, !"maxntidx", i32 256}

template <int max_threads_per_block>
__global__ void
__launch_bounds__(max_threads_per_block)
Kernel3()
{
}

template void Kernel3<MAX_THREADS_PER_BLOCK>();
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel3{{.*}}, !"maxntidx", i32 256}

template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp)
Kernel4()
{
}
template void Kernel4<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();

// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel4{{.*}}, !"maxntidx", i32 256}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel4{{.*}}, !"minctasm", i32 2}

const int constint = 100;
template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block + constint,
                  min_blocks_per_mp + max_threads_per_block)
Kernel5()
{
}
template void Kernel5<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();

// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel5{{.*}}, !"maxntidx", i32 356}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel5{{.*}}, !"minctasm", i32 258}

// Make sure we don't emit negative launch bounds values.
__global__ void
__launch_bounds__( -MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel6()
{
}
// CHECK-NOT: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel6{{.*}}, !"maxntidx",
// CHECK:     !{{[0-9]+}} = !{void ()* @{{.*}}Kernel6{{.*}}, !"minctasm",

__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP )
Kernel7()
{
}
// CHECK:     !{{[0-9]+}} = !{void ()* @{{.*}}Kernel7{{.*}}, !"maxntidx",
// CHECK-NOT: !{{[0-9]+}} = !{void ()* @{{.*}}Kernel7{{.*}}, !"minctasm",

const char constchar = 12;
__global__ void __launch_bounds__(constint, constchar) Kernel8() {}
// CHECK:     !{{[0-9]+}} = !{void ()* @{{.*}}Kernel8{{.*}}, !"maxntidx", i32 100
// CHECK:     !{{[0-9]+}} = !{void ()* @{{.*}}Kernel8{{.*}}, !"minctasm", i32 12
