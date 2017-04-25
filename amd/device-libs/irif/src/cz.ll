; ===--------------------------------------------------------------------------
;                    ROCm Device Libraries
; 
;  This file is distributed under the University of Illinois Open Source
;  License. See LICENSE.TXT for details.
; ===--------------------------------------------------------------------------

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

declare i8 @llvm.ctlz.i8(i8, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)

declare i8 @llvm.cttz.i8(i8, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)

define i32 @__llvm_ctlz_i32(i32) #0 {
    %2 = call i32 @llvm.ctlz.i32(i32 %0, i1 1)
    ret i32 %2
}

define i64 @__llvm_ctlz_i64(i64) #0 {
    %2 = call i64 @llvm.ctlz.i64(i64 %0, i1 1)
    ret i64 %2
}

define i32 @__llvm_cttz_i32(i32) #0 {
    %2 = call i32 @llvm.cttz.i32(i32 %0, i1 1)
    ret i32 %2
}

define i64 @__llvm_cttz_i64(i64) #0 {
    %2 = call i64 @llvm.cttz.i64(i64 %0, i1 1)
    ret i64 %2
}

attributes #0 = { alwaysinline argmemonly norecurse nounwind readnone }

