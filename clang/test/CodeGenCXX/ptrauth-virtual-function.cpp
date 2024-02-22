// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - | FileCheck %s

// Check virtual function pointers in vtables are signed and their relocation
// structures are emitted.

// CHECK: %[[CLASS_B1:.*]] = type { ptr }

// CHECK: @_ZTV2B1 = unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI2B1, ptr @_ZN2B12m0Ev.ptrauth] }, align 8
// CHECK: @_ZTV2B1.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV2B1, i32 0, inrange i32 0, i32 2), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"
// CHECK: @g_B1 = global %[[CLASS_B1]] { ptr @_ZTV2B1.ptrauth }

// CHECK: @_ZTV2B0 = unnamed_addr constant { [7 x ptr] } { [7 x ptr] [{{.*}} ptr @_ZN2B02m0Ev.ptrauth, ptr @_ZN2B02m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth, ptr @_ZN2B0D1Ev.ptrauth, ptr @_ZN2B0D0Ev.ptrauth] }
// CHECK: @_ZN2B02m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2B0D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2B0D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D0 = unnamed_addr constant { [9 x ptr] } { [9 x ptr] [{{.*}} ptr @_ZN2D02m0Ev.ptrauth, ptr @_ZTch0_h4_N2D02m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth.1, ptr @_ZN2D0D1Ev.ptrauth, ptr @_ZN2D0D0Ev.ptrauth, ptr @_ZN2D02m1Ev.ptrauth, ptr @_ZN2D02m3Ev.ptrauth] }
// CHECK: @_ZN2D02m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D02m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTch0_h4_N2D02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.1 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D0D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D0D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D02m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 7) to i64), i64 35045 }, section "llvm.ptrauth"
// CHECK: @_ZN2D02m3Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D02m3Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, i32 0, i32 8) to i64), i64 10565 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D1 = unnamed_addr constant { [8 x ptr] } { [8 x ptr] [{{.*}} ptr @_ZN2D12m0Ev.ptrauth, ptr @_ZTch0_h4_N2D12m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth.2, ptr @_ZN2D1D1Ev.ptrauth, ptr @_ZN2D1D0Ev.ptrauth, ptr @_ZN2D12m1Ev.ptrauth] }
// CHECK: @_ZN2D12m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D12m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D12m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTch0_h4_N2D12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.2 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D1D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D1D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D1D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D1D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D12m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, i32 0, i32 7) to i64), i64 52864 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D2 = unnamed_addr constant { [9 x ptr], [8 x ptr] } { [9 x ptr] [{{.*}} ptr @_ZN2D22m0Ev.ptrauth, ptr @_ZTch0_h4_N2D22m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth.3, ptr @_ZN2D2D1Ev.ptrauth, ptr @_ZN2D2D0Ev.ptrauth, ptr @_ZN2D22m1Ev.ptrauth, ptr @_ZN2D22m3Ev.ptrauth], [8 x ptr] [{{.*}} ptr @_ZThn16_N2D22m0Ev.ptrauth, ptr @_ZTchn16_h4_N2D22m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth.4, ptr @_ZThn16_N2D2D1Ev.ptrauth, ptr @_ZThn16_N2D2D0Ev.ptrauth, ptr @_ZThn16_N2D22m1Ev.ptrauth] }
// CHECK: @_ZN2D22m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D22m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D22m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTch0_h4_N2D22m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.3 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D2D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D2D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D2D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D2D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D22m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D22m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 7) to i64), i64 35045 }, section "llvm.ptrauth"
// CHECK: @_ZN2D22m3Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D22m3Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 8) to i64), i64 10565 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D22m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D22m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTchn16_h4_N2D22m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTchn16_h4_N2D22m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.4 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D2D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D2D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D2D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D2D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D22m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D22m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, i32 1, i32 7) to i64), i64 52864 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D3 = unnamed_addr constant { [7 x ptr], [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 32 to ptr), ptr null, ptr @_ZTI2D3, ptr @_ZN2D32m0Ev.ptrauth, ptr @_ZN2D32m1Ev.ptrauth, ptr @_ZN2D3D1Ev.ptrauth, ptr @_ZN2D3D0Ev.ptrauth], [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI2D3, ptr @_ZThn16_N2D32m0Ev.ptrauth, ptr @_ZThn16_N2D32m1Ev.ptrauth, ptr @_ZThn16_N2D3D1Ev.ptrauth, ptr @_ZThn16_N2D3D0Ev.ptrauth], [11 x ptr] [ptr inttoptr (i64 -32 to ptr), ptr null, ptr inttoptr (i64 -32 to ptr), ptr inttoptr (i64 -32 to ptr), ptr inttoptr (i64 -32 to ptr), ptr @_ZTI2D3, ptr @_ZTv0_n24_N2D32m0Ev.ptrauth, ptr @_ZTcv0_n32_h4_N2D32m1Ev.ptrauth, ptr @_ZN2B02m2Ev.ptrauth.11, ptr @_ZTv0_n48_N2D3D1Ev.ptrauth, ptr @_ZTv0_n48_N2D3D0Ev.ptrauth] }

// CHECK: @_ZTC2D30_2V0 = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 32 to ptr), ptr null, ptr @_ZTI2V0, ptr @_ZN2V02m0Ev.ptrauth.12, ptr @_ZN2V02m1Ev.ptrauth.13, ptr @_ZN2V0D1Ev.ptrauth.14, ptr @_ZN2V0D0Ev.ptrauth.15], [11 x ptr] [ptr inttoptr (i64 -32 to ptr), ptr null, ptr inttoptr (i64 -32 to ptr), ptr inttoptr (i64 -32 to ptr), ptr inttoptr (i64 -32 to ptr), ptr @_ZTI2V0, ptr @_ZTv0_n24_N2V02m0Ev.ptrauth.16, ptr @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth.17, ptr @_ZN2B02m2Ev.ptrauth.18, ptr @_ZTv0_n48_N2V0D1Ev.ptrauth.19, ptr @_ZTv0_n48_N2V0D0Ev.ptrauth.20] }

// CHECK: @_ZN2V02m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V02m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n24_N2V02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTcv0_n32_h4_N2V02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.5 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"

// CHECK: @_ZTC2D316_2V1 = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI2V1, ptr @_ZN2V12m0Ev.ptrauth.21, ptr @_ZN2V12m1Ev.ptrauth.22, ptr @_ZN2V1D1Ev.ptrauth.23, ptr @_ZN2V1D0Ev.ptrauth.24], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI2V1, ptr @_ZTv0_n24_N2V12m0Ev.ptrauth.25, ptr @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth.26, ptr @_ZN2B02m2Ev.ptrauth.27, ptr @_ZTv0_n48_N2V1D1Ev.ptrauth.28, ptr @_ZTv0_n48_N2V1D0Ev.ptrauth.29] }
// CHECK: @_ZN2V12m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V12m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V1D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V1D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V12m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n24_N2V12m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTcv0_n32_h4_N2V12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.6 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V1D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V1D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D32m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D32m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2D32m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D32m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2D3D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D3D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2D3D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN2D3D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D32m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D32m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 1, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D32m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D32m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 1, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D3D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D3D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 1, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D3D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn16_N2D3D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 1, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2D32m0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n24_N2D32m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 2, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2D32m1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTcv0_n32_h4_N2D32m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 2, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.11 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 2, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2D3D1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2D3D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 2, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2D3D0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2D3D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [7 x ptr], [11 x ptr] }, ptr @_ZTV2D3, i32 0, i32 2, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m0Ev.ptrauth.12 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m1Ev.ptrauth.13 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D1Ev.ptrauth.14 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D0Ev.ptrauth.15 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V02m0Ev.ptrauth.16 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n24_N2V02m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth.17 = private constant { ptr, i32, i64, i64 } { ptr @_ZTcv0_n32_h4_N2V02m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.18 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D1Ev.ptrauth.19 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V0D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D0Ev.ptrauth.20 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V0D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D30_2V0, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m0Ev.ptrauth.21 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V12m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m1Ev.ptrauth.22 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D1Ev.ptrauth.23 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V1D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D0Ev.ptrauth.24 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2V1D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 0, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V12m0Ev.ptrauth.25 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n24_N2V12m0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth.26 = private constant { ptr, i32, i64, i64 } { ptr @_ZTcv0_n32_h4_N2V12m1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.27 = private constant { ptr, i32, i64, i64 } { ptr @_ZN2B02m2Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D1Ev.ptrauth.28 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V1D1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D0Ev.ptrauth.29 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N2V1D0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC2D316_2V1, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"


struct S0 {
  int f;
};

struct S1 {
  int f;
};

struct S2 : S0, S1 {
  int f;
};

class B0 {
public:
  virtual void m0();
  virtual S1 *m1();
  virtual void m2();
  virtual ~B0();
  int f;
};

class B1 {
public:
  virtual void m0();
};

class D0 : public B0 {
public:
  void m0() override;
  S2 *m1() override;
  virtual void m3();
  int f;
};

class D1 : public B0 {
public:
  void m0() override;
  S2 *m1() override;
  int f;
};

class D2 : public D0, public D1 {
public:
  void m0() override;
  S2 *m1() override;
  void m3() override;
  int f;
};

class V0 : public virtual B0 {
public:
  void m0() override;
  S2 *m1() override;
  int f;
};

class V1 : public virtual B0 {
public:
  void m0() override;
  S2 *m1() override;
  ~V1();
  int f;
};

class D3 : public V0, public V1 {
public:
  void m0() override;
  S2 *m1() override;
  int f;
};

B1 g_B1;

void B0::m0() {}

void B1::m0() {}

void D0::m0() {}

void D1::m0() {}

void D2::m0() {}

void D3::m0() {}

V1::~V1() {
  m1();
}

// Check sign/authentication of vtable pointers and authentication of virtual
// functions.

// CHECK-LABEL: define noundef ptr @_ZN2V1D2Ev(
// CHECK: %[[THIS1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T1:[0-9]+]] = ptrtoint ptr %[[T0]] to i64
// CHECK: %[[T2:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T1]], i32 2, i64 0)
// CHECK: %[[T3:[0-9]+]] = inttoptr i64 %[[T2]] to ptr
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[T3]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 %[[T6]], i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T7]] to ptr
// CHECK: store ptr %[[SIGNED_VTADDR]], ptr %[[THIS1]]

// CHECK-LABEL: define void @_Z8testB0m0P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m0(B0 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testB0m1P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 1
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 15165)
// CHECK: call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m1(B0 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testB0m2P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m2(B0 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z8testD0m0P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m0(D0 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD0m1P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 35045)
// CHECK: call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m1(D0 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testD0m2P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m2(D0 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z8testD0m3P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 6
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 10565)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m3(D0 *a) {
  a->m3();
}


// CHECK-LABEL: define void @_Z8testD1m0P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m0(D1 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD1m1P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 52864)
// CHECK: call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m1(D1 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testD1m2P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m2(D1 *a) {
  a->m2();
}


// CHECK-LABEL: define void @_Z8testD2m0P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(36) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m0(D2 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD2m1P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 35045)
// CHECK: call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(36) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m1(D2 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z10testD2m2D0P2D2(
// CHECK: call void @_ZN2B02m2Ev(ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}){{$}}

void testD2m2D0(D2 *a) {
  a->D0::m2();
}

// CHECK-LABEL: define void @_Z10testD2m2D1P2D2(
// CHECK: call void @_ZN2B02m2Ev(ptr noundef nonnull align {{[0-9]+}} dereferenceable(12) %{{.*}}){{$}}

void testD2m2D1(D2 *a) {
  a->D1::m2();
}

// CHECK-LABEL: define void @_Z8testD2m3P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 6
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 10565)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(36) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m3(D2 *a) {
  a->m3();
}

// CHECK-LABEL: define void @_Z8testD3m0P2D3(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 44578)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(32) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3m0(D3 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD3m1P2D3(
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 1
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 30766)
// CHECK: call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(32) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3m1(D3 *a) {
  a->m1();
}

// CHECK: define void @_Z8testD3m2P2D3(ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %[[V0]], align 8
// CHECK: %[[V1:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V1]], i32 2, i64 0)
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to ptr
// CHECK: %[[VBASE_OFFSET_PTR:.*]] = getelementptr i8, ptr %[[V3]], i64 -24
// CHECK: %[[VBASE_OFFSET:.*]] = load i64, ptr %[[VBASE_OFFSET_PTR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 %[[VBASE_OFFSET]]
// CHECK: %[[VTABLE1:.*]] = load ptr, ptr %[[ADD_PTR]], align 8
// CHECK: %[[V4:.*]] = ptrtoint ptr %[[VTABLE1]] to i64
// CHECK: %[[V5:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V4]], i32 2, i64 0)
// CHECK: %[[V6:.*]] = inttoptr i64 %[[V5]] to ptr
// CHECK: %[[VFN:.*]] = getelementptr inbounds ptr, ptr %[[V6]], i64 2
// CHECK: %[[V7:.*]] = load ptr, ptr %[[VFN]], align 8
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V8]], i64 43073)
// CHECK: call void %[[V7]](ptr noundef nonnull align 8 dereferenceable(12) %[[ADD_PTR]]) [ "ptrauth"(i32 0, i64 %[[V9]]) ]

void testD3m2(D3 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z17testD3Destructor0P2D3(
// CHECK: load ptr, ptr
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[T2:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 3
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 62452)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(32) %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3Destructor0(D3 *a) {
  delete a;
}

// CHECK-LABEL: define void @_Z17testD3Destructor1P2D3(
// CHECK: %[[T6:.*]] = load ptr, ptr %
// CHECK: %[[VTABLE0:[a-z0-9]+]] = load ptr, ptr %
// CHECK: %[[T2:[0-9]+]] = ptrtoint ptr %[[VTABLE0]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[COMPLETE_OFFSET_PTR:.*]] = getelementptr inbounds i64, ptr %[[T4]], i64 -2
// CHECK: %[[T5:[0-9]+]] = load i64, ptr %[[COMPLETE_OFFSET_PTR]]
// CHECK: %[[T7:[0-9]+]] = getelementptr inbounds i8, ptr %[[T6]], i64 %[[T5]]
// CHECK: %[[VTABLE1:[a-z0-9]+]] = load ptr, ptr %[[T6]]
// CHECK: %[[T9:[0-9]+]] = ptrtoint ptr %[[VTABLE1]] to i64
// CHECK: %[[T10:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T9]], i32 2, i64 0)
// CHECK: %[[T11:[0-9]+]] = inttoptr i64 %[[T10]] to ptr
// CHECK: %[[VFN:[a-z0-9]+]] = getelementptr inbounds ptr, ptr %[[T11]], i64 2
// CHECK: %[[T12:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T13:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T14:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T13]], i64 57279)
// CHECK: %call = call noundef ptr %[[T12]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(32) %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T14]]) ]
// CHECK: call void @_ZdlPv(ptr noundef %[[T7]])

void testD3Destructor1(D3 *a) {
  ::delete a;
}

// CHECK-LABEL: define void @_Z17testD3Destructor2P2D3(
// CHECK: load ptr, ptr
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %
// CHECK: %[[T2:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:.*]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:.*]] = getelementptr inbounds ptr, ptr %[[T4]], i64 2
// CHECK: %[[T5:.*]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:.*]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 57279)
// CHECK: %call = call noundef ptr %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(32) %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3Destructor2(D3 *a) {
  a->~D3();
}

void materializeConstructors() {
  B0 B0;
  B1 B1;
  D0 D0;
  D1 D1;
  D2 D2;
  D3 D3;
  V0 V0;
  V1 V1;
}

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZN2B0C2Ev(
// CHECK: %[[THIS:.*]] = load ptr, ptr %
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV2B0, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to ptr
// CHECK: store ptr %[[SIGNED_VTADDR]], ptr %[[THIS]]

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZN2D0C2Ev(
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr] }, ptr @_ZTV2D0, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to ptr
// CHECK: store ptr %[[SIGNED_VTADDR]], ptr %[[THIS]]

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZN2D1C2Ev(
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr getelementptr inbounds ({ [8 x ptr] }, ptr @_ZTV2D1, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to ptr
// CHECK: store ptr %[[SIGNED_VTADDR]], ptr %[[THIS]]

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZN2D2C2Ev(
// CHECK: %[[SLOT0:.*]] = load ptr, ptr
// CHECK: %[[SIGN_VTADDR0:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[T1:[0-9]+]] = inttoptr i64 %[[SIGN_VTADDR0]] to ptr
// CHECK: store ptr %[[T1]], ptr %[[SLOT0]]
// CHECK: %[[T3:[a-z0-9.]+]] = getelementptr inbounds i8, ptr %[[SLOT0]], i64 16
// CHECK: %[[SIGN_VTADDR1:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr getelementptr inbounds ({ [9 x ptr], [8 x ptr] }, ptr @_ZTV2D2, i32 0, inrange i32 1, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[T5:[0-9]+]] = inttoptr i64 %[[SIGN_VTADDR1]] to ptr
// CHECK: store ptr %[[T5]], ptr %[[T3]]

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZN2V0C2Ev(
// CHECK: %[[THIS1]] = load ptr, ptr %
// CHECK: %[[VTT:[a-z0-9]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = load ptr, ptr %[[VTT]]
// CHECK: %[[T1:[0-9]+]] = ptrtoint ptr %[[T0]] to i64
// CHECK: %[[T2:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T1]], i32 2, i64 0)
// CHECK: %[[T3:[0-9]+]] = inttoptr i64 %[[T2]] to ptr
// CHECK: %[[VTADDR0:[0-9]+]] = ptrtoint ptr %[[T3]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 %[[VTADDR0]], i32 2, i64 0)
// CHECK: %[[SIGN_VTADDR0:[0-9]+]] = inttoptr i64 %[[T7]] to ptr
// CHECK: store ptr %[[SIGN_VTADDR0]], ptr %[[SLOT0]]
// CHECK: %[[T9:[0-9]+]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// CHECK: %[[T10:[0-9]+]] = load ptr, ptr %[[T9]]
// CHECK: %[[T11:[0-9]+]] = ptrtoint ptr %[[T10]] to i64
// CHECK: %[[T12:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T11]], i32 2, i64 0)
// CHECK: %[[T13:[0-9]+]] = inttoptr i64 %[[T12]] to ptr
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %[[THIS1]]
// CHECK: %[[T15:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T16:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T15]], i32 2, i64 0)
// CHECK: %[[T17:[0-9]+]] = inttoptr i64 %[[T16]] to ptr
// CHECK: %[[VBASE_OFFSET_PTR:[a-z.]+]] = getelementptr i8, ptr %[[T17]], i64 -24
// CHECK: %[[VBASE_OFFSET:[a-z.]+]] = load i64, ptr %[[VBASE_OFFSET_PTR]]
// CHECK: %[[T20:[a-z.]+]] = getelementptr inbounds i8, ptr %[[THIS1]], i64 %[[VBASE_OFFSET]]
// CHECK: %[[VTADDR1:[0-9]+]] = ptrtoint ptr %[[T13]] to i64
// CHECK: %[[T23:[0-9]+]] = call i64 @llvm.ptrauth.sign(i64 %[[VTADDR1]], i32 2, i64 0)
// CHECK: %[[SIGN_VTADDR1:[0-9]+]] = inttoptr i64 %[[T23]] to ptr
// CHECK: store ptr %[[SIGN_VTADDR1]], ptr %[[T20]]
