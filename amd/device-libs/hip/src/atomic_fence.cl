#include "ockl.h"
#include "irif.h"

#define ATTR2 __attribute__((always_inline))
ATTR2 void
__atomic_work_item_fence(cl_mem_fence_flags flags, memory_order order, memory_scope scope)
{
    // We're tying global-happens-before and local-happens-before together as does HSA
    if (order != memory_order_relaxed) {
        switch (scope) {
        case memory_scope_work_item:
            break;
        case memory_scope_sub_group:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_sg(); break;
            case memory_order_release: __llvm_fence_rel_sg(); break;
            case memory_order_acq_rel: __llvm_fence_ar_sg(); break;
            case memory_order_seq_cst: __llvm_fence_sc_sg(); break;
            }
            break;
        case memory_scope_work_group:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_wg(); break;
            case memory_order_release: __llvm_fence_rel_wg(); break;
            case memory_order_acq_rel: __llvm_fence_ar_wg(); break;
            case memory_order_seq_cst: __llvm_fence_sc_wg(); break;
            }
            break;
        case memory_scope_device:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_dev(); break;
            case memory_order_release: __llvm_fence_rel_dev(); break;
            case memory_order_acq_rel: __llvm_fence_ar_dev(); break;
            case memory_order_seq_cst: __llvm_fence_sc_dev(); break;
            }
            break;
        case memory_scope_all_svm_devices:
            switch (order) {
            case memory_order_relaxed: break;
            case memory_order_acquire: __llvm_fence_acq_sys(); break;
            case memory_order_release: __llvm_fence_rel_sys(); break;
            case memory_order_acq_rel: __llvm_fence_ar_sys(); break;
            case memory_order_seq_cst: __llvm_fence_sc_sys(); break;
            }
            break;
        }
    }
}
