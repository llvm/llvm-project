; ===--------------------------------------------------------------------------
;                    ROCm Device Libraries
; 
;  This file is distributed under the University of Illinois Open Source
;  License. See LICENSE.TXT for details.
; ===------------------------------------------------------------------------*/

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.read_register.i32(metadata) #0
declare i64 @llvm.read_register.i64(metadata) #0

define i64 @__llvm_amdgcn_read_exec() #1 {
    %1 = call i64 @llvm.read_register.i64(metadata !0) #0
    ret i64 %1
}

define i32 @__llvm_amdgcn_read_exec_lo() #1 {
    %1 = call i32 @llvm.read_register.i32(metadata !1) #0
    ret i32 %1
}

define i32 @__llvm_amdgcn_read_exec_hi() #1 {
    %1 = call i32 @llvm.read_register.i32(metadata !2) #0
    ret i32 %1
}

attributes #0 = { nounwind convergent }
attributes #1 = { alwaysinline nounwind convergent }

!0 = !{!"exec"}
!1 = !{!"exec_lo"}
!2 = !{!"exec_hi"}
