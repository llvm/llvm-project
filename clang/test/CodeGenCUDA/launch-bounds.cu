// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -target-cpu sm_90 -DUSE_MAX_BLOCKS -fcuda-is-device -emit-llvm -o - | FileCheck -check-prefix=CHECK_MAX_BLOCKS %s

#include "Inputs/cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2
#ifdef USE_MAX_BLOCKS
#define MAX_BLOCKS_PER_MP     4
#endif

// CHECK: @Kernel1() #[[ATTR0:[0-9]+]]
// CHECK: @Kernel2() #[[ATTR1:[0-9]+]]
// CHECK: @{{.*}}Kernel3{{.*}}() #[[ATTR1]]
// CHECK: @{{.*}}Kernel4{{.*}}() #[[ATTR0]]
// CHECK: @{{.*}}Kernel5{{.*}}() #[[ATTR2:[0-9]+]]
// CHECK: @{{.*}}Kernel6{{.*}}() #[[ATTR3:[0-9]+]]
// CHECK: @{{.*}}Kernel7{{.*}}() #[[ATTR1]]
// CHECK: @{{.*}}Kernel8{{.*}}() #[[ATTR4:[0-9]+]]

// CHECK-DAG: attributes #[[ATTR0]] = {{{.*}} "nvvm.maxntid"="256" "nvvm.minctasm"="2" {{.*}}}
// CHECK-DAG: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxntid"="256" {{.*}}}
// CHECK-DAG: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="356" "nvvm.minctasm"="258" {{.*}}}
// CHECK-DAG: attributes #[[ATTR3]] = {{{.*}} "nvvm.minctasm"="2" {{.*}}}
// CHECK-DAG: attributes #[[ATTR4]] = {{{.*}} "nvvm.maxntid"="100" "nvvm.minctasm"="12" {{.*}}}

// CHECK_MAX_BLOCKS: @Kernel1_sm_90() #[[ATTR0:[0-9]+]]
// CHECK_MAX_BLOCKS: @{{.*}}Kernel4_sm_90{{.*}} #[[ATTR0]]
// CHECK_MAX_BLOCKS: @{{.*}}Kernel5_sm_90{{.*}} #[[ATTR1:[0-9]+]]
// CHECK_MAX_BLOCKS: @{{.*}}Kernel7_sm_90{{.*}} #[[ATTR2:[0-9]+]]
// CHECK_MAX_BLOCKS: @{{.*}}Kernel8_sm_90{{.*}} #[[ATTR3:[0-9]+]]

// CHECK_MAX_BLOCKS-DAG: attributes #[[ATTR0]] = {{{.*}} "nvvm.maxclusterrank"="4" "nvvm.maxntid"="256" "nvvm.minctasm"="2" {{.*}}}
// CHECK_MAX_BLOCKS-DAG: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxclusterrank"="260" "nvvm.maxntid"="356" "nvvm.minctasm"="258" {{.*}}}
// CHECK_MAX_BLOCKS-DAG: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="256" {{.*}}}
// CHECK_MAX_BLOCKS-DAG: attributes #[[ATTR3]] = {{{.*}} "nvvm.maxclusterrank"="14" "nvvm.maxntid"="100" "nvvm.minctasm"="12" {{.*}}}

// Test both max threads per block and Min cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel1()
{
}
}

#ifdef USE_MAX_BLOCKS
// Test max threads per block and min/max cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP )
Kernel1_sm_90()
{
}
}
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

template <int max_threads_per_block>
__global__ void
__launch_bounds__(max_threads_per_block)
Kernel3()
{
}

template __global__ void Kernel3<MAX_THREADS_PER_BLOCK>();

template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp)
Kernel4()
{
}
template __global__ void Kernel4<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();


#ifdef USE_MAX_BLOCKS
template <int max_threads_per_block, int min_blocks_per_mp, int max_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp, max_blocks_per_mp)
Kernel4_sm_90()
{
}
template __global__ void Kernel4_sm_90<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP>();

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

#endif //USE_MAX_BLOCKS

// Make sure we don't emit negative launch bounds values.
__global__ void
__launch_bounds__( -MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel6()
{
}

__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP )
Kernel7()
{
}

#ifdef USE_MAX_BLOCKS
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP, -MAX_BLOCKS_PER_MP )
Kernel7_sm_90()
{
}
#endif // USE_MAX_BLOCKS

const char constchar = 12;
__global__ void __launch_bounds__(constint, constchar) Kernel8() {}

#ifdef USE_MAX_BLOCKS
const char constchar_2 = 14;
__global__ void __launch_bounds__(constint, constchar, constchar_2) Kernel8_sm_90() {}
#endif // USE_MAX_BLOCKS
