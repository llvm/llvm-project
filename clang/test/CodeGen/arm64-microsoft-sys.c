// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -S \
// RUN: -o - %s | FileCheck %s -check-prefix CHECK-ASM

// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -emit-llvm \
// RUN: -o - %s | FileCheck %s -check-prefix CHECK-IR

// RUN: %clang_cc1 -triple arm64-darwin -fms-compatibility -emit-llvm \
// RUN: -o - %s | FileCheck %s -check-prefix CHECK-IR

// From winnt.h
// op0=1 encodings, use with __sys
#define ARM64_SYSINSTR(op0, op1, crn, crm, op2) \
        ( ((op1 & 7) << 11) | \
          ((crn & 15) << 7) | \
          ((crm & 15) << 3) | \
          ((op2 & 7) << 0) )

//
// Sampling of instructions
//
#define ARM64_DC_CGDSW_EL1      ARM64_SYSINSTR(1,0, 7,10,6) // Clean of Data and Allocation Tags by Set/Way
#define ARM64_IC_IALLU_EL1      ARM64_SYSINSTR(1,0, 7, 5,0) // Instruction Cache Invalidate All to PoU
#define ARM64_AT_S1E2W          ARM64_SYSINSTR(1,4, 7, 8,1) // Translate Stage1, EL2, write
#define ARM64_TLBI_VMALLE1      ARM64_SYSINSTR(1,0, 8, 7,0) // Invalidate stage 1 TLB [CP15_TLBIALL]
#define ARM64_CFP_RCTX          ARM64_SYSINSTR(1,3, 7, 3,4) // Control Flow Prediction Restriction by Context

// From intrin.h
unsigned int __sys(int, __int64);

void check__sys(__int64 v) {
  __int64 ret;

  __sys(ARM64_DC_CGDSW_EL1, v);
// CHECK-ASM: sys     #0, c7, c10, #6, x8
// CHECK-IR: %[[VAR:.*]] = load i64,
// CHECK-IR-NEXT: call void @llvm.aarch64.sys(i32 0, i32 7, i32 10, i32 6, i64 %[[VAR]])

  __sys(ARM64_IC_IALLU_EL1, v);
// CHECK-ASM: sys     #0, c7, c5, #0, x8
// CHECK-IR: %[[VAR:.*]] = load i64,
// CHECK-IR-NEXT: call void @llvm.aarch64.sys(i32 0, i32 7, i32 5, i32 0, i64 %[[VAR]])

  __sys(ARM64_AT_S1E2W, v);
// CHECK-ASM: at      s1e2w, x8
// CHECK-IR: %[[VAR:.*]] = load i64,
// CHECK-IR-NEXT: call void @llvm.aarch64.sys(i32 4, i32 7, i32 8, i32 1, i64 %[[VAR]])

  __sys(ARM64_TLBI_VMALLE1, v);
// CHECK-ASM: sys     #0, c8, c7, #0, x8
// CHECK-IR: %[[VAR:.*]] = load i64,
// CHECK-IR-NEXT: call void @llvm.aarch64.sys(i32 0, i32 8, i32 7, i32 0, i64 %[[VAR]])

  __sys(ARM64_CFP_RCTX, v);
// CHECK-ASM: sys     #3, c7, c3, #4, x8
// CHECK-IR: %[[VAR:.*]] = load i64,
// CHECK-IR-NEXT: call void @llvm.aarch64.sys(i32 3, i32 7, i32 3, i32 4, i64 %[[VAR]])
}
