target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

;;;;; Load
define i32 @__llvm_ld_atomic_a1_x_dev_i32(i32 addrspace(1)* nocapture readonly) #1 {
  %2 = load atomic i32, i32 addrspace(1)* %0 monotonic, align 4
    ret i32 %2
}


;;;;; Add
define i32 @__llvm_atomic_add_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile add i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_add_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #0 {
  %3 = atomicrmw volatile add i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define i32 @__llvm_atomic_add_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #0 {
  %3 = atomicrmw volatile add i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_add_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #0 {
  %3 = atomicrmw volatile add i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}


;;;;; Max
define i32 @__llvm_atomic_max_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile max i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define i32 @__llvm_atomic_umax_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile umax i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_max_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #0 {
  %3 = atomicrmw volatile max i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define i64 @__llvm_atomic_umax_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #0 {
  %3 = atomicrmw volatile umax i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define i32 @__llvm_atomic_max_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #0 {
  %3 = atomicrmw volatile max i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define i32 @__llvm_atomic_umax_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #0 {
  %3 = atomicrmw volatile umax i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_max_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #0 {
  %3 = atomicrmw volatile max i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

define i64 @__llvm_atomic_umax_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #0 {
  %3 = atomicrmw volatile umax i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; Min
define i32 @__llvm_atomic_min_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile min i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define i32 @__llvm_atomic_umin_a1_x_dev_i32(i32 addrspace(1)* nocapture, i32) #0 {
  %3 = atomicrmw volatile umin i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_min_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #0 {
  %3 = atomicrmw volatile min i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define i64 @__llvm_atomic_umin_a1_x_dev_i64(i64 addrspace(1)* nocapture, i64) #0 {
  %3 = atomicrmw volatile umin i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %3
}

define i32 @__llvm_atomic_min_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #0 {
  %3 = atomicrmw volatile min i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define i32 @__llvm_atomic_umin_a3_x_wg_i32(i32 addrspace(3)* nocapture, i32) #0 {
  %3 = atomicrmw volatile umin i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %3
}

define i64 @__llvm_atomic_min_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #0 {
  %3 = atomicrmw volatile min i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

define i64 @__llvm_atomic_umin_a3_x_wg_i64(i64 addrspace(3)* nocapture, i64) #0 {
  %3 = atomicrmw volatile umin i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %3
}

;;;;; cmpxchg
define i32 @__llvm_cmpxchg_a1_x_x_dev_i32(i32 addrspace(1)* nocapture, i32, i32) #0 {
  %4 = cmpxchg volatile i32 addrspace(1)* %0, i32 %1, i32 %2 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

define i64 @__llvm_cmpxchg_a1_x_x_dev_i64(i64 addrspace(1)* nocapture, i64, i64) #0 {
  %4 = cmpxchg volatile i64 addrspace(1)* %0, i64 %1, i64 %2 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 0
  ret i64 %5
}

define i32 @__llvm_cmpxchg_a3_x_x_wg_i32(i32 addrspace(3)* nocapture, i32, i32) #0 {
  %4 = cmpxchg volatile i32 addrspace(3)* %0, i32 %1, i32 %2 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

define i64 @__llvm_cmpxchg_a3_x_x_wg(i64 addrspace(3)* nocapture, i64, i64) #0 {
  %4 = cmpxchg volatile i64 addrspace(3)* %0, i64 %1, i64 %2 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 0
  ret i64 %5
}


attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { alwaysinline norecurse nounwind readonly }

