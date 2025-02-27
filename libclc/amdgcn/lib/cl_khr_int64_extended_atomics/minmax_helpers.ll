define i64 @__clc__sync_fetch_and_min_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umin_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_min_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umin_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_max_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umax_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_max_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umax_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}
