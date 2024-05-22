// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -target-cpu sm_90 -DUSE_MAX_BLOCKS -fcuda-is-device -emit-llvm -o - | FileCheck -check-prefix=CHECK_MAX_BLOCKS %s

#include "Inputs/cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2
#ifdef USE_MAX_BLOCKS
#define MAX_BLOCKS_PER_MP     4
#endif

// Test both max threads per block and Min cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel1()
{
}
}

// CHECK: !{{[0-9]+}} = !{ptr @Kernel1, !"maxntidx", i32 256}
// CHECK: !{{[0-9]+}} = !{ptr @Kernel1, !"minctasm", i32 2}

#ifdef USE_MAX_BLOCKS
// Test max threads per block and min/max cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP )
Kernel1_sm_90()
{
}
}

// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @Kernel1_sm_90, !"maxntidx", i32 256}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @Kernel1_sm_90, !"minctasm", i32 2}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @Kernel1_sm_90, !"maxclusterrank", i32 4}
#endif // USE_MAX_BLOCKS

// Test only max threads per block. Min cta per sm defaults to 0, and
// CodeGen doesn't output a zero value for minctasm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK )
Kernel2()
{
}
}

// CHECK: !{{[0-9]+}} = !{ptr @Kernel2, !"maxntidx", i32 256}

template <int max_threads_per_block>
__global__ void
__launch_bounds__(max_threads_per_block)
Kernel3()
{
}

template __global__ void Kernel3<MAX_THREADS_PER_BLOCK>();
// CHECK: !{{[0-9]+}} = !{ptr @{{.*}}Kernel3{{.*}}, !"maxntidx", i32 256}

template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp)
Kernel4()
{
}
template __global__ void Kernel4<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();

// CHECK: !{{[0-9]+}} = !{ptr @{{.*}}Kernel4{{.*}}, !"maxntidx", i32 256}
// CHECK: !{{[0-9]+}} = !{ptr @{{.*}}Kernel4{{.*}}, !"minctasm", i32 2}

#ifdef USE_MAX_BLOCKS
template <int max_threads_per_block, int min_blocks_per_mp, int max_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp, max_blocks_per_mp)
Kernel4_sm_90()
{
}
template __global__ void Kernel4_sm_90<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP>();

// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel4_sm_90{{.*}}, !"maxntidx", i32 256}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel4_sm_90{{.*}}, !"minctasm", i32 2}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel4_sm_90{{.*}}, !"maxclusterrank", i32 4}
#endif //USE_MAX_BLOCKS

const int constint = 100;
template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block + constint,
                  min_blocks_per_mp + max_threads_per_block)
Kernel5()
{
}
template __global__ void Kernel5<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();

// CHECK: !{{[0-9]+}} = !{ptr @{{.*}}Kernel5{{.*}}, !"maxntidx", i32 356}
// CHECK: !{{[0-9]+}} = !{ptr @{{.*}}Kernel5{{.*}}, !"minctasm", i32 258}

#ifdef USE_MAX_BLOCKS

template <int max_threads_per_block, int min_blocks_per_mp, int max_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block + constint,
                  min_blocks_per_mp + max_threads_per_block,
                  max_blocks_per_mp + max_threads_per_block)
Kernel5_sm_90()
{
}
template __global__ void Kernel5_sm_90<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP>();

// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel5_sm_90{{.*}}, !"maxntidx", i32 356}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel5_sm_90{{.*}}, !"minctasm", i32 258}
// CHECK_MAX_BLOCKS: !{{[0-9]+}} = !{ptr @{{.*}}Kernel5_sm_90{{.*}}, !"maxclusterrank", i32 260}
#endif //USE_MAX_BLOCKS

// Make sure we don't emit negative launch bounds values.
__global__ void
__launch_bounds__( -MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel6()
{
}
// CHECK-NOT: !{{[0-9]+}} = !{ptr @{{.*}}Kernel6{{.*}}, !"maxntidx",
// CHECK:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel6{{.*}}, !"minctasm",

__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP )
Kernel7()
{
}
// CHECK:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel7{{.*}}, !"maxntidx",
// CHECK-NOT: !{{[0-9]+}} = !{ptr @{{.*}}Kernel7{{.*}}, !"minctasm",

#ifdef USE_MAX_BLOCKS
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP, -MAX_BLOCKS_PER_MP )
Kernel7_sm_90()
{
}
// CHECK_MAX_BLOCKS:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel7_sm_90{{.*}}, !"maxntidx",
// CHECK_MAX_BLOCKS-NOT: !{{[0-9]+}} = !{ptr @{{.*}}Kernel7_sm_90{{.*}}, !"minctasm",
// CHECK_MAX_BLOCKS-NOT: !{{[0-9]+}} = !{ptr @{{.*}}Kernel7_sm_90{{.*}}, !"maxclusterrank",
#endif // USE_MAX_BLOCKS

const char constchar = 12;
__global__ void __launch_bounds__(constint, constchar) Kernel8() {}
// CHECK:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel8{{.*}}, !"maxntidx", i32 100
// CHECK:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel8{{.*}}, !"minctasm", i32 12

#ifdef USE_MAX_BLOCKS
const char constchar_2 = 14;
__global__ void __launch_bounds__(constint, constchar, constchar_2) Kernel8_sm_90() {}
// CHECK_MAX_BLOCKS:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel8_sm_90{{.*}}, !"maxntidx", i32 100
// CHECK_MAX_BLOCKS:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel8_sm_90{{.*}}, !"minctasm", i32 12
// CHECK_MAX_BLOCKS:     !{{[0-9]+}} = !{ptr @{{.*}}Kernel8_sm_90{{.*}}, !"maxclusterrank", i32 14
#endif // USE_MAX_BLOCKS
