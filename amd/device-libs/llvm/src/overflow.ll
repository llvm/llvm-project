; ===--------------------------------------------------------------------------
;                    ROCm Device Libraries
; 
;  This file is distributed under the University of Illinois Open Source
;  License. See LICENSE.TXT for details.
; ===------------------------------------------------------------------------*/

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

declare {i16, i1} @llvm.sadd.with.overflow.i16(i16, i16)
declare {i16, i1} @llvm.uadd.with.overflow.i16(i16, i16)

define zeroext i1 @__llvm_sadd_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_uadd_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.uadd.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)

define zeroext i1 @__llvm_sadd_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_uadd_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

declare {i64, i1} @llvm.sadd.with.overflow.i64(i64, i64)
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64)

define zeroext i1 @__llvm_sadd_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_uadd_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

declare {i16, i1} @llvm.ssub.with.overflow.i16(i16, i16)
declare {i16, i1} @llvm.usub.with.overflow.i16(i16, i16)

define zeroext i1 @__llvm_ssub_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.ssub.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_usub_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.usub.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32)

define zeroext i1 @__llvm_ssub_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_usub_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64)
declare {i64, i1} @llvm.usub.with.overflow.i64(i64, i64)

define zeroext i1 @__llvm_ssub_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_usub_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

declare {i16, i1} @llvm.smul.with.overflow.i16(i16, i16)
declare {i16, i1} @llvm.umul.with.overflow.i16(i16, i16)

define zeroext i1 @__llvm_smul_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_umul_with_overflow_i16(i16, i16, i16* nocapture) #0 {
  %4 = call {i16, i1} @llvm.umul.with.overflow.i16(i16 %0, i16 %1)
  %5 = extractvalue {i16, i1} %4, 0
  store i16 %5, i16* %2, align 4
  %6 = extractvalue {i16, i1} %4, 1
  ret i1 %6
}

declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32)

define zeroext i1 @__llvm_smul_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_umul_with_overflow_i32(i32, i32, i32* nocapture) #0 {
  %4 = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %0, i32 %1)
  %5 = extractvalue {i32, i1} %4, 0
  store i32 %5, i32* %2, align 4
  %6 = extractvalue {i32, i1} %4, 1
  ret i1 %6
}

declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64)
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)

define zeroext i1 @__llvm_smul_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

define zeroext i1 @__llvm_umul_with_overflow_i64(i64, i64, i64* nocapture) #0 {
  %4 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %0, i64 %1)
  %5 = extractvalue {i64, i1} %4, 0
  store i64 %5, i64* %2, align 4
  %6 = extractvalue {i64, i1} %4, 1
  ret i1 %6
}

attributes #0 = { alwaysinline argmemonly norecurse nounwind }

