// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.cir.ll

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ogcg.ll

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda -fclangir \
// RUN:            -target-cpu sm_90 -DUSE_MAX_BLOCKS \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.max.cir
// RUN: FileCheck --check-prefix=CIR_MAX_BLOCKS %s --input-file=%t.max.cir

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda -fclangir \
// RUN:            -target-cpu sm_90 -DUSE_MAX_BLOCKS \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.max.cir.ll
// RUN: FileCheck --check-prefix=LLVM_MAX_BLOCKS %s --input-file=%t.max.cir.ll

// RUN: %clang_cc1 -triple nvptx-unknown-unknown -x cuda \
// RUN:            -target-cpu sm_90 -DUSE_MAX_BLOCKS \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.max.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM_MAX_BLOCKS %s --input-file=%t.max.ogcg.ll

#include "Inputs/cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2
#ifdef USE_MAX_BLOCKS
#define MAX_BLOCKS_PER_MP     4
#endif

// LLVM: @Kernel1() #[[ATTR0:[0-9]+]]
// LLVM: @Kernel2() #[[ATTR1:[0-9]+]]
// LLVM: @{{.*}}Kernel3{{.*}}() #[[ATTR1]]
// LLVM: @{{.*}}Kernel4{{.*}}() #[[ATTR0]]
// LLVM: @{{.*}}Kernel5{{.*}}() #[[ATTR2:[0-9]+]]
// LLVM: @{{.*}}Kernel6{{.*}}() #[[ATTR3:[0-9]+]]
// LLVM: @{{.*}}Kernel7{{.*}}() #[[ATTR1]]
// LLVM: @{{.*}}Kernel8{{.*}}() #[[ATTR4:[0-9]+]]

// LLVM-DAG: attributes #[[ATTR0]] = {{{.*}} "nvvm.maxntid"="256" "nvvm.minctasm"="2" {{.*}}}
// LLVM-DAG: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxntid"="256" {{.*}}}
// LLVM-DAG: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="356" "nvvm.minctasm"="258" {{.*}}}
// LLVM-DAG: attributes #[[ATTR3]] = {{{.*}} "nvvm.minctasm"="2" {{.*}}}
// LLVM-DAG: attributes #[[ATTR4]] = {{{.*}} "nvvm.maxntid"="100" "nvvm.minctasm"="12" {{.*}}}

// LLVM_MAX_BLOCKS: @Kernel1_sm_90() #[[ATTR0:[0-9]+]]
// LLVM_MAX_BLOCKS: @{{.*}}Kernel4_sm_90{{.*}} #[[ATTR0]]
// LLVM_MAX_BLOCKS: @{{.*}}Kernel5_sm_90{{.*}} #[[ATTR1:[0-9]+]]
// LLVM_MAX_BLOCKS: @{{.*}}Kernel7_sm_90{{.*}} #[[ATTR2:[0-9]+]]
// LLVM_MAX_BLOCKS: @{{.*}}Kernel8_sm_90{{.*}} #[[ATTR3:[0-9]+]]

// LLVM_MAX_BLOCKS-DAG: attributes #[[ATTR0]] = {{{.*}} "nvvm.maxclusterrank"="4" "nvvm.maxntid"="256" "nvvm.minctasm"="2" {{.*}}}
// LLVM_MAX_BLOCKS-DAG: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxclusterrank"="260" "nvvm.maxntid"="356" "nvvm.minctasm"="258" {{.*}}}
// LLVM_MAX_BLOCKS-DAG: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="256" {{.*}}}
// LLVM_MAX_BLOCKS-DAG: attributes #[[ATTR3]] = {{{.*}} "nvvm.maxclusterrank"="14" "nvvm.maxntid"="100" "nvvm.minctasm"="12" {{.*}}}

// Test both max threads per block and Min cta per sm.
// CIR: cir.func {{.*}} @Kernel1() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"{{.*}}cir.nvvm.minctasm = "2"
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel1()
{
}
}

#ifdef USE_MAX_BLOCKS
// Test max threads per block and min/max cta per sm.
// CIR_MAX_BLOCKS: cir.func {{.*}} @Kernel1_sm_90() cc(ptx_kernel){{.*}}cir.nvvm.maxclusterrank = "4"{{.*}}cir.nvvm.maxntid = "256"{{.*}}cir.nvvm.minctasm = "2"
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
// CIR: cir.func {{.*}} @Kernel2() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"
// CIR-NOT: cir.nvvm.minctasm
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK )
Kernel2()
{
}
}

// CIR: cir.func {{.*}} @_Z7Kernel3ILi256EEvv() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"
template <int max_threads_per_block>
__global__ void
__launch_bounds__(max_threads_per_block)
Kernel3()
{
}

template __global__ void Kernel3<MAX_THREADS_PER_BLOCK>();

// CIR: cir.func {{.*}} @_Z7Kernel4ILi256ELi2EEvv() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"{{.*}}cir.nvvm.minctasm = "2"
template <int max_threads_per_block, int min_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp)
Kernel4()
{
}
template __global__ void Kernel4<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP>();


#ifdef USE_MAX_BLOCKS
// CIR_MAX_BLOCKS: cir.func {{.*}} @_Z13Kernel4_sm_90ILi256ELi2ELi4EEvv() cc(ptx_kernel){{.*}}cir.nvvm.maxclusterrank = "4"{{.*}}cir.nvvm.maxntid = "256"{{.*}}cir.nvvm.minctasm = "2"
template <int max_threads_per_block, int min_blocks_per_mp, int max_blocks_per_mp>
__global__ void
__launch_bounds__(max_threads_per_block, min_blocks_per_mp, max_blocks_per_mp)
Kernel4_sm_90()
{
}
template __global__ void Kernel4_sm_90<MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP, MAX_BLOCKS_PER_MP>();

#endif //USE_MAX_BLOCKS

// CIR: cir.func {{.*}} @_Z7Kernel5ILi256ELi2EEvv() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "356"{{.*}}cir.nvvm.minctasm = "258"
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

// CIR_MAX_BLOCKS: cir.func {{.*}} @_Z13Kernel5_sm_90ILi256ELi2ELi4EEvv() cc(ptx_kernel){{.*}}cir.nvvm.maxclusterrank = "260"{{.*}}cir.nvvm.maxntid = "356"{{.*}}cir.nvvm.minctasm = "258"
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
// CIR: cir.func {{.*}} @_Z7Kernel6v() cc(ptx_kernel){{.*}}cir.nvvm.minctasm = "2"
// CIR-NOT: cir.nvvm.maxntid
__global__ void
__launch_bounds__( -MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel6()
{
}

// CIR: cir.func {{.*}} @_Z7Kernel7v() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"
// CIR-NOT: cir.nvvm.minctasm
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP )
Kernel7()
{
}

#ifdef USE_MAX_BLOCKS
// CIR_MAX_BLOCKS: cir.func {{.*}} @_Z13Kernel7_sm_90v() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "256"
// CIR_MAX_BLOCKS-NOT: cir.nvvm.maxclusterrank
// CIR_MAX_BLOCKS-NOT: cir.nvvm.minctasm
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, -MIN_BLOCKS_PER_MP, -MAX_BLOCKS_PER_MP )
Kernel7_sm_90()
{
}
#endif // USE_MAX_BLOCKS

// CIR: cir.func {{.*}} @_Z7Kernel8v() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "100"{{.*}}cir.nvvm.minctasm = "12"
// CIR_MAX_BLOCKS: cir.func {{.*}} @_Z7Kernel8v() cc(ptx_kernel){{.*}}cir.nvvm.maxntid = "100"{{.*}}cir.nvvm.minctasm = "12"
const char constchar = 12;
__global__ void __launch_bounds__(constint, constchar) Kernel8() {}

#ifdef USE_MAX_BLOCKS
// CIR_MAX_BLOCKS: cir.func {{.*}} @_Z13Kernel8_sm_90v() cc(ptx_kernel){{.*}}cir.nvvm.maxclusterrank = "14"{{.*}}cir.nvvm.maxntid = "100"{{.*}}cir.nvvm.minctasm = "12"
const char constchar_2 = 14;
__global__ void __launch_bounds__(constint, constchar, constchar_2) Kernel8_sm_90() {}
#endif // USE_MAX_BLOCKS
