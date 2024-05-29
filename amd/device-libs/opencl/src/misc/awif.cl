/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__attribute__((overloadable)) void
mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_acq_rel, memory_scope_work_group);
}

__attribute__((overloadable)) void
read_mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_acquire, memory_scope_work_group);
}

__attribute__((overloadable)) void
write_mem_fence(cl_mem_fence_flags flags)
{
    atomic_work_item_fence(flags, memory_order_release, memory_scope_work_group);
}

#define IMPL_ATOMIC_WORK_ITEM_FENCE(...)                                                                        \
    if (order != memory_order_relaxed) {                                                                        \
        switch (scope) {                                                                                        \
        case memory_scope_work_item:                                                                            \
            break;                                                                                              \
        case memory_scope_sub_group:                                                                            \
            switch (order) {                                                                                    \
            case memory_order_relaxed: break;                                                                   \
            case memory_order_acquire: __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront" __VA_ARGS__); break;\
            case memory_order_release: __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront" __VA_ARGS__); break;\
            case memory_order_acq_rel: __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront" __VA_ARGS__); break;\
            case memory_order_seq_cst: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront" __VA_ARGS__); break;\
            }                                                                                                   \
            break;                                                                                              \
        case memory_scope_work_group:                                                                           \
            switch (order) {                                                                                    \
            case memory_order_relaxed: break;                                                                   \
            case memory_order_acquire: __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup" __VA_ARGS__); break;\
            case memory_order_release: __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup" __VA_ARGS__); break;\
            case memory_order_acq_rel: __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup" __VA_ARGS__); break;\
            case memory_order_seq_cst: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup" __VA_ARGS__); break;\
            }                                                                                                   \
            break;                                                                                              \
        case memory_scope_device:                                                                               \
            switch (order) {                                                                                    \
            case memory_order_relaxed: break;                                                                   \
            case memory_order_acquire: __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent" __VA_ARGS__); break;    \
            case memory_order_release: __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent" __VA_ARGS__); break;    \
            case memory_order_acq_rel: __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent" __VA_ARGS__); break;    \
            case memory_order_seq_cst: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent" __VA_ARGS__); break;    \
            }                                                                                                   \
            break;                                                                                              \
        case memory_scope_all_svm_devices:                                                                      \
            switch (order) {                                                                                    \
            case memory_order_relaxed: break;                                                                   \
            case memory_order_acquire: __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "" __VA_ARGS__); break;         \
            case memory_order_release: __builtin_amdgcn_fence(__ATOMIC_RELEASE, "" __VA_ARGS__); break;         \
            case memory_order_acq_rel: __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "" __VA_ARGS__); break;         \
            case memory_order_seq_cst: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "" __VA_ARGS__); break;         \
            }                                                                                                   \
            break;                                                                                              \
        }                                                                                                       \
    }

__attribute__((overloadable)) void
atomic_work_item_fence(cl_mem_fence_flags flags, memory_order order, memory_scope scope)
{
    // The AS to fence (if only global or local is needed) is encoded in
    // metadata attached to the fence instruction by the builtin.
    // That metadata may be dropped in some cases, if that happens then
    // we are tying global-happens-before and local-happens-before together
    // as does HSA

    if (flags) {
        // global or image is set, but not local -> fence only global memory.
        if ((flags & CLK_LOCAL_MEM_FENCE) == 0) {
            IMPL_ATOMIC_WORK_ITEM_FENCE(, "global");
            return;
        }

        // only local is set
        if (flags == CLK_LOCAL_MEM_FENCE) {
            IMPL_ATOMIC_WORK_ITEM_FENCE(, "local");
            return;
        }

        // all flags are set, same as if none are set -> fence all.
    }

    IMPL_ATOMIC_WORK_ITEM_FENCE();
}
