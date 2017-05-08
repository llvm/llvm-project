; ModuleID = 'hc_amdgcn.bc'

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

define i64 @__activelanemask_v4_b64_b1(i32 %input) #5 {
  %a = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %input) #9
  ret i64 %a
}

define i32 @__amdgcn_wave_sr1(i32 %v, i1 %b) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 312, i32 15, i32 15, i1 %b)
  ret i32 %call
}

define i32 @__amdgcn_wave_sl1(i32 %v, i1 %b) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 304, i32 15, i32 15, i1 %b)
  ret i32 %call
}

define i32 @__amdgcn_wave_rr1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 316, i32 15, i32 15, i1 0)
  ret i32 %call
}

define i32 @__amdgcn_wave_rl1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 308, i32 15, i32 15, i1 0)
  ret i32 %call
}

define i32 @__amdgcn_row_rshift(i32 %data, i32 %delta) #3 {
  switch i32 %delta, label %31 [
    i32 1, label %1
    i32 2, label %3
    i32 3, label %5
    i32 4, label %7
    i32 5, label %9
    i32 6, label %11
    i32 7, label %13
    i32 8, label %15
    i32 9, label %17
    i32 10, label %19
    i32 11, label %21
    i32 12, label %23
    i32 13, label %25
    i32 14, label %27
    i32 15, label %29
  ]

; <label>:1:                                              ; preds = %0                     
  %2 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 273, i32 15, i32 15, i1 0)
  ret i32 %2

; <label>:3:                                              ; preds = %0                    
  %4 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 274, i32 15, i32 15, i1 0)
  ret i32 %4

; <label>:5:                                              ; preds = %0                     
  %6 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 275, i32 15, i32 15, i1 0)
  ret i32 %6

; <label>:7:                                              ; preds = %0                     
  %8 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 276, i32 15, i32 15, i1 0)
  ret i32 %8

; <label>:9:                                              ; preds = %0                     
  %10 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 277, i32 15, i32 15, i1 0)
  ret i32 %10

; <label>:11:                                              ; preds = %0                     
  %12 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 278, i32 15, i32 15, i1 0)
  ret i32 %12

; <label>:13:                                              ; preds = %0                     
  %14 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 279, i32 15, i32 15, i1 0)
  ret i32 %14

; <label>:15:                                              ; preds = %0                     
  %16 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 280, i32 15, i32 15, i1 0)
  ret i32 %16

; <label>:17:                                              ; preds = %0                     
  %18 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 281, i32 15, i32 15, i1 0)
  ret i32 %18

; <label>:19:                                              ; preds = %0                     
  %20 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 282, i32 15, i32 15, i1 0)
  ret i32 %20

; <label>:21:                                              ; preds = %0                     
  %22 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 283, i32 15, i32 15, i1 0)
  ret i32 %22

; <label>:23:                                              ; preds = %0                     
  %24 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 284, i32 15, i32 15, i1 0)
  ret i32 %24

; <label>:25:                                              ; preds = %0                     
  %26 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 285, i32 15, i32 15, i1 0)
  ret i32 %26

; <label>:27:                                              ; preds = %0                     
  %28 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 286, i32 15, i32 15, i1 0)
  ret i32 %28

; <label>:29:                                              ; preds = %0                     
  %30 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 287, i32 15, i32 15, i1 0)
  ret i32 %30

; <label>:31:
  ret i32 %data
}

define i32 @__amdgcn_ds_permute(i32 %index, i32 %src) #3  {
  %call = call i32 @llvm.amdgcn.ds.permute(i32 %index, i32 %src)
  ret i32 %call
}

;llvm.amdgcn.ds.permute <index> <src>
declare i32 @llvm.amdgcn.ds.permute(i32, i32) #4

define i32 @__amdgcn_ds_bpermute(i32 %index, i32 %src) #3  {
  %call = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src)
  ret i32 %call
}

;llvm.amdgcn.ds.bpermute <index> <src>
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

define i32 @__amdgcn_ds_swizzle(i32 %src, i32 %pattern) #3  {
  %call = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 %pattern)
  ret i32 %call
}

;llvm.amdgcn.ds.swizzle <index> <src>
declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #4

define i32 @__amdgcn_move_dpp(i32 %src, i32 %dpp_ctrl, i32 %row_mask, i32 %bank_mask, i1 %bound_ctrl) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %src, i32 %dpp_ctrl, i32 %row_mask, i32 %bank_mask, i1 %bound_ctrl)
  ret i32 %call
}

; llvm.amdgcn.mov.dpp.i32 <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1) #4

define i32 @__atomic_wrapinc_global(i32 addrspace(1)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.inc.i32.p1i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32, i32, i1) #4

; Function Attrs: nounwind argmemonly
define i32 @__atomic_wrapinc_local(i32 addrspace(3)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.inc.i32.p3i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32, i32, i1) #4

define i32 @__atomic_wrapinc(i32 addrspace(4)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.inc.i32.p4i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* nocapture, i32, i32, i32, i1) #4

define i32 @__atomic_wrapdec_global(i32 addrspace(1)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.dec.i32.p1i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32, i32, i1) #4

define i32 @__atomic_wrapdec_local(i32 addrspace(3)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.dec.i32.p3i32(i32 addrspace(3)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.dec.i32.p3i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.dec.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32, i32, i1) #4

define i32 @__atomic_wrapdec(i32 addrspace(4)* nocapture %addr, i32 %val) #1 {
  %ret = tail call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* nocapture %addr, i32 %val, i32 7, i32 1, i1 0)
  ret i32 %ret
}

; llvm.amdgcn.atomic.dec.i32.p4i32 <addr> <val> <ordering> <scope> <is_volatile>
declare i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* nocapture, i32, i32, i32, i1) #4

define i64 @__clock_u64() #1 {
  %ret = tail call i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %ret
}

declare i64 @llvm.amdgcn.s.memrealtime() #1


define i64 @__cycle_u64() #1 {
  %ret = tail call i64 @llvm.amdgcn.s.memtime()
  ret i64 %ret
}

declare i64 @llvm.amdgcn.s.memtime() #1

define i32 @get_group_segment_size() #0 {
  %1 = call i32 @llvm.amdgcn.s.getreg(i32 17158) #0
  %2 = shl nuw nsw i32 %1, 8 ; from 64 dwords to bytes
  ret i32 %2
}

define i8 addrspace(4)* @get_group_segment_base_pointer() #0 {
  ; XXX For some reason getreg may return strange values for LDS_BASE
  ; temporary fix as 0 for now
 
  ;%1 = call i32 @llvm.amdgcn.s.getreg(i32 14342) #0
  %1 = add i32 0, 0
  %2 = shl nuw nsw i32 %1, 8 ; from 64 dwords to bytes

  ; make it a pointer to LDS first...
  %3 = inttoptr i32 %2 to i8 addrspace(3)*

  ; then convert to generic address space
  %4 = addrspacecast i8 addrspace(3)* %3 to i8 addrspace(4)*
  ret i8 addrspace(4)* %4
}

define i32 @get_static_group_segment_size() #1 {
  %ret = call i32 @llvm.amdgcn.groupstaticsize() #1
  ret i32 %ret
}

define i8 addrspace(4)* @get_dynamic_group_segment_base_pointer() #0 {
  %1 = tail call i8 addrspace(4)* @get_group_segment_base_pointer() #0
  %2 = tail call i32 @get_static_group_segment_size() #1
  %3 = zext i32 %2 to i64
  %4 = getelementptr inbounds i8, i8 addrspace(4)* %1, i64 %3
  ret i8 addrspace(4)* %4
}

declare i32 @llvm.amdgcn.s.getreg(i32) #0

declare i32 @llvm.amdgcn.groupstaticsize() #1

attributes #0 = { alwaysinline nounwind readonly }
attributes #1 = { alwaysinline nounwind readnone }
attributes #3 = { alwaysinline convergent nounwind }
attributes #4 = { convergent nounwind }
attributes #5 = { alwaysinline nounwind }
attributes #6 = { alwaysinline norecurse nounwind readnone }
attributes #7 = { norecurse nounwind readnone }
attributes #9 = { convergent nounwind readnone }
