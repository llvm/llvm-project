; ModuleID = 'hc_atomic.bc'

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_unsigned(i32 addrspace(4)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(4)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_exchange_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_int_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_int_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_compare_exchange_int(i32 addrspace(4)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg volatile i32 addrspace(4)* %x, i32 %y, i32 %z seq_cst monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_add_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_sub_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define float @atomic_exchange_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg volatile i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define float @atomic_exchange_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg volatile i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define float @atomic_exchange_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg volatile i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define float @atomic_add_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg volatile i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define float @atomic_add_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg volatile i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define float @atomic_add_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg volatile i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define float @atomic_sub_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg volatile i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define float @atomic_sub_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg volatile i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define float @atomic_sub_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg volatile i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_exchange_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xchg i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_compare_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg volatile i64 addrspace(1)* %x, i64 %y, i64 %z seq_cst monotonic
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_compare_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg volatile i64 addrspace(3)* %x, i64 %y, i64 %z seq_cst monotonic
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_compare_exchange_uint64(i64 addrspace(4)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg volatile i64 addrspace(4)* %x, i64 %y, i64 %z seq_cst monotonic
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_add_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile add i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_add_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile add i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_add_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile add i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(1)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(3)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_and_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile and i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_or_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile or i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_xor_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_max_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile max i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_min_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw volatile min i32 addrspace(4)* %x, i32 %y seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_and_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile and i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_or_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile or i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_xor_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_max_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile max i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_min_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile min i64 addrspace(1)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_and_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile and i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_or_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile or i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_xor_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_max_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile max i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_min_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile min i64 addrspace(3)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_and_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile and i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_or_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile or i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_xor_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile xor i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_max_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile max i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i64 @atomic_min_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw volatile min i64 addrspace(4)* %x, i64 %y seq_cst
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_unsigned_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(1)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_unsigned_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(1)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_unsigned_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(3)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_unsigned_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(3)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_unsigned(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(4)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_unsigned(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(4)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_int_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(1)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_int_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(1)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_int_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(3)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_int_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(3)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_inc_int(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw volatile add i32 addrspace(4)* %x, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @atomic_dec_int(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw volatile sub i32 addrspace(4)* %x, i32 1 seq_cst
  ret i32 %ret
}
attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { alwaysinline nounwind readnone }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { nounwind }
