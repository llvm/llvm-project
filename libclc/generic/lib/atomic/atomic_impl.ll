define i32 @__clc_atomic_add_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_add_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_cmpxchg_addr1(i32 addrspace(1)* nocapture %ptr, i32 %compare, i32 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i32 addrspace(1)* %ptr, i32 %compare, i32 %value seq_cst seq_cst
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

define i32 @__clc_atomic_cmpxchg_addr3(i32 addrspace(3)* nocapture %ptr, i32 %compare, i32 %value) nounwind alwaysinline {
entry:
  %0 = cmpxchg volatile i32 addrspace(3)* %ptr, i32 %compare, i32 %value seq_cst seq_cst
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

define i32 @__clc_atomic_max_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_max_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_min_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_min_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_or_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_or_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile or i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umin_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umin_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xchg_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xchg_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xchg i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xor_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_xor_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile xor i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}
