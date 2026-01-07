#ifndef __SPIRVINTRIN_H
#define __SPIRVINTRIN_H

#ifndef __SPIRV__
#error "This file is intended for SPIRV targets or offloading to SPIRV"
#endif

#ifndef __GPUINTRIN_H
#error "Never use <spirvintrin.h> directly; include <gpuintrin.h> instead"
#endif

#include <stdint.h>
#if !defined(__cplusplus)
_Pragma("push_macro(\"bool\")");
#define bool _Bool
#define true 1
#define false 0
#endif

_Pragma("omp begin declare target device_type(nohost)");
_Pragma("omp begin declare variant match(device = {arch(spirv64)})");

// Type aliases to the address spaces used by the SPIR-V backend.
//
#define __gpu_private  __attribute__((address_space(0)))
#define __gpu_constant
#define __gpu_local
#define __gpu_global __attribute__((address_space(1)))
#define __gpu_generic __attribute__((address_space(4)))
// Attribute to declare a function as a kernel.
#define __gpu_kernel __attribute__((spirv_kernel, visibility("protected")))
#define __SPIRV_VAR_QUALIFIERS extern const
// Workgroup and invocation ID functions
uint64_t __spirv_BuiltInNumWorkgroups(int i);
uint64_t __spirv_BuiltInWorkgroupId(int i);
uint64_t __spirv_BuiltInWorkgroupSize(int i);
uint64_t __spirv_BuiltInLocalInvocationId(int i);

typedef enum {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4
} Scope_t;

typedef enum {
  Relaxed = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10
} MemorySemantics_t;

using unsigned ProgramAS = 9;

#ifdef __cplusplus
template <typename... Args>
int __spirv_ocl_printf(Args...);
#endif

// Subgroup
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupLocalInvocationId;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupSize;

// Group non-uniform operations
uint64_t __spirv_GroupNonUniformBallot(uint32_t execution_scope, bool predicate);
uint32_t __spirv_GroupNonUniformBroadcastFirst(uint32_t execution_scope, uint32_t value);
uint32_t __spirv_GroupNonUniformShuffle(uint32_t execution_scope, uint32_t value, uint32_t id);

// Synchronization
void __spirv_ControlBarrier(uint32_t execution_scope, uint32_t memory_scope, uint32_t semantics);
void __spirv_MemoryBarrier(uint32_t memory_scope, uint32_t semantics);

// Atomic
uint32_t __spirv_AtomicIAdd(uint32_t *, int, int, uint32_t);
void __spirv_AtomicStore(int32_t *, int, int, int);
int32_t __spirv_AtomicLoad(int32_t *, int, int);
int32_t __spirv_AtomicCompareExchange(int32_t *, int, int, int, int, int);


// Returns the number of blocks in the 'x' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_x(void) {
   return __spirv_BuiltInNumWorkgroups(0);
}

// Returns the number of blocks in the 'y' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_y(void) {
   return __spirv_BuiltInNumWorkgroups(1);
}

// Returns the number of blocks in the 'z' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_z(void) {
   return __spirv_BuiltInNumWorkgroups(2);
}

// Returns the 'x' dimension of the current block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_x(void) {
  return __spirv_BuiltInWorkgroupId(0);
}

// Returns the 'y' dimension of the current block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_y(void) {
  return __spirv_BuiltInWorkgroupId(1);
}

// Returns the 'z' dimension of the current block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_z(void) {
  return __spirv_BuiltInWorkgroupId(2);
}

// Returns the number of threads in the 'x' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_x(void) {
  return __spirv_BuiltInWorkgroupSize(0);
}

// Returns the number of threads in the 'y' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_y(void) {
  return __spirv_BuiltInWorkgroupSize(1);
}

// Returns the number of threads in the 'z' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_z(void) {
  return __spirv_BuiltInWorkgroupSize(2);
}

// Returns the 'x' dimension id of the thread in the current block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_x(void) {
  return __spirv_BuiltInLocalInvocationId(0);
}

// Returns the 'y' dimension id of the thread in the current block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_y(void) {
  return __spirv_BuiltInLocalInvocationId(1);
}

// Returns the 'z' dimension id of the thread in the current block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_z(void) {
  return __spirv_BuiltInLocalInvocationId(2);
}

// Returns the size of a warp, always 32 on NVIDIA hardware.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_lanes(void) {
  return __spirv_BuiltInSubgroupSize;
}

// Returns the id of the thread inside of a warp executing together.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_lane_id(void) {
  return __spirv_BuiltInSubgroupLocalInvocationId;
}
 
// Returns the bit-mask of active threads in the current warp.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_lane_mask(void) { 
  uint32_t Size = __gpu_num_lanes();
  return ((uint64_t)1 << Size) - (uint64_t)1;
}
// Copies the value from the first active thread in the warp to the rest.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_read_first_lane_u32(uint64_t __lane_mask, uint32_t __x) {
  return __spirv_GroupNonUniformBroadcastFirst(3, __x);
}
// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_ballot(uint64_t __lane_mask,
                                                          bool __x) {
  uint64_t ballot = __spirv_GroupNonUniformBallot(3, __x);
  return __lane_mask & ballot;
}
// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_threads(void) {
   __spirv_ControlBarrier(Scope_t::Workgroup, Scope_t::Workgroup, 
      0x100 | MemorySemantics_t::SequentiallyConsistent);
}
// Waits for all threads in the warp to reconverge for independent scheduling.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_lane(uint64_t __lane_mask) {
   __spirv_ControlBarrier(Scope_t::Subgroup, Scope_t::Subgroup, 
      0x80 | MemorySemantics_t::SequentiallyConsistent);
}
// Shuffles the the lanes inside the warp according to the given index.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_shuffle_idx_u32(uint64_t __lane_mask, uint32_t __idx, uint32_t __x,
                      uint32_t __width) {
  uint32_t __lane = __idx + (__gpu_lane_id() & ~(__width - 1));
  return __spirv_GroupNonUniformShuffle(3, __x, __lane);
}

// Returns a bitmask marking all lanes that have the same value of __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u32(uint64_t __lane_mask, uint32_t __x) {
  return __gpu_match_any_u32_impl(__lane_mask, __x);
}

// Returns a bitmask marking all lanes that have the same value of __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u64(uint64_t __lane_mask, uint64_t __x) {
  return __gpu_match_any_u64_impl(__lane_mask, __x);
}

// Returns the current lane mask if every lane contains __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u32(uint64_t __lane_mask, uint32_t __x) {
  return __gpu_match_all_u32_impl(__lane_mask, __x);
}


// Returns the current lane mask if every lane contains __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u64(uint64_t __lane_mask, uint64_t __x) {
  return __gpu_match_all_u64_impl(__lane_mask, __x);
}

// Returns true if the flat pointer points to 'shared' memory.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_local(void *ptr) {
  return false; // TODO
  //return to_local(ptr) != 0;
}
// Returns true if the flat pointer points to 'local' memory.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_private(void *ptr) {
  return false;
  //return to_private(ptr) != 0; // TODO
}
// Terminates execution of the calling thread.
_DEFAULT_FN_ATTRS [[noreturn]] static __inline__ void __gpu_exit(void) {
}
// Suspend the thread briefly to assist the scheduler during busy loops.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_thread_suspend(void) {
}

_Pragma("omp end declare variant");
_Pragma("omp end declare target");

#if !defined(__cplusplus)
_Pragma("pop_macro(\"bool\")");
#endif
#endif // __SPIRVINTRIN_H
