// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - | FileCheck %s

// Check virtual function pointers in vtables are signed and their relocation
// structures are emitted.

// CHECK: @_ZTV2B1 = unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI2B1 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B12m0Ev.ptrauth to i8*)] }, align 8
// CHECK: @_ZTV2B1.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV2B1, i32 0, inrange i32 0, i32 2) to i8*), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"
// CHECK: @g_B1 = global { i8** } { i8** getelementptr inbounds ({ i8*, i32, i64, i64 }, { i8*, i32, i64, i64 }* @_ZTV2B1.ptrauth, i32 0, i32 0) }

// CHECK: @_ZTV2B0 = unnamed_addr constant { [7 x i8*] } { [7 x i8*] [{{.*}} i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B0D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B0D0Ev.ptrauth to i8*)] }
// CHECK: @_ZN2B02m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S1* (%class.B0*)* @_ZN2B02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2B0D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.B0* (%class.B0*)* @_ZN2B0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2B0D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D0 = unnamed_addr constant { [9 x i8*] } { [9 x i8*] [{{.*}} i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D02m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTch0_h4_N2D02m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.1 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D0D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D0D0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D02m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D02m3Ev.ptrauth to i8*)] }
// CHECK: @_ZN2D02m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D0*)* @_ZN2D02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D02m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D0*)* @_ZTch0_h4_N2D02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.1 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D0D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D0* (%class.D0*)* @_ZN2D0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D0D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D0*)* @_ZN2D0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D02m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D0*)* @_ZN2D02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 7) to i64), i64 35045 }, section "llvm.ptrauth"
// CHECK: @_ZN2D02m3Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D0*)* @_ZN2D02m3Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, i32 0, i32 8) to i64), i64 10565 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D1 = unnamed_addr constant { [8 x i8*] } { [8 x i8*] [{{.*}} i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D12m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTch0_h4_N2D12m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.2 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D1D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D1D0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D12m1Ev.ptrauth to i8*)] }
// CHECK: @_ZN2D12m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D1*)* @_ZN2D12m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D12m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D1*)* @_ZTch0_h4_N2D12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.2 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D1D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D1* (%class.D1*)* @_ZN2D1D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D1D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D1*)* @_ZN2D1D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D12m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D1*)* @_ZN2D12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, i32 0, i32 7) to i64), i64 52864 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D2 = unnamed_addr constant { [9 x i8*], [8 x i8*] } { [9 x i8*] [{{.*}} i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D22m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTch0_h4_N2D22m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.3 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D2D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D2D0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D22m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D22m3Ev.ptrauth to i8*)], [8 x i8*] [{{.*}} i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D22m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTchn16_h4_N2D22m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.4 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D2D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D2D0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D22m1Ev.ptrauth to i8*)] }
// CHECK: @_ZN2D22m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D2*)* @_ZN2D22m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTch0_h4_N2D22m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D2*)* @_ZTch0_h4_N2D22m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.3 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZN2D2D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D2* (%class.D2*)* @_ZN2D2D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZN2D2D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D2*)* @_ZN2D2D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D22m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D2*)* @_ZN2D22m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 7) to i64), i64 35045 }, section "llvm.ptrauth"
// CHECK: @_ZN2D22m3Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D2*)* @_ZN2D22m3Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 0, i32 8) to i64), i64 10565 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D22m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D2*)* @_ZThn16_N2D22m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 2) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTchn16_h4_N2D22m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D2*)* @_ZTchn16_h4_N2D22m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 3) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.4 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 4) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D2D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D2* (%class.D2*)* @_ZThn16_N2D2D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 5) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D2D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D2*)* @_ZThn16_N2D2D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 6) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D22m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D2*)* @_ZThn16_N2D22m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, i32 1, i32 7) to i64), i64 52864 }, section "llvm.ptrauth"

// CHECK: @_ZTV2D3 = unnamed_addr constant { [7 x i8*], [7 x i8*], [11 x i8*] } { [7 x i8*] [i8* inttoptr (i64 32 to i8*), i8* null, i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }* @_ZTI2D3 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D32m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D32m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D3D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2D3D0Ev.ptrauth to i8*)], [7 x i8*] [i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -16 to i8*), i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }* @_ZTI2D3 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D32m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D32m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D3D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZThn16_N2D3D0Ev.ptrauth to i8*)], [11 x i8*] [i8* inttoptr (i64 -32 to i8*), i8* null, i8* inttoptr (i64 -32 to i8*), i8* inttoptr (i64 -32 to i8*), i8* inttoptr (i64 -32 to i8*), i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }* @_ZTI2D3 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n24_N2D32m0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTcv0_n32_h4_N2D32m1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.11 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2D3D1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2D3D0Ev.ptrauth to i8*)] }

// CHECK: @_ZTC2D30_2V0 = unnamed_addr constant { [7 x i8*], [11 x i8*] } { [7 x i8*] [i8* inttoptr (i64 32 to i8*), i8* null, i8* bitcast (i8** @_ZTI2V0 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V02m0Ev.ptrauth.12 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V02m1Ev.ptrauth.13 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V0D1Ev.ptrauth.14 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V0D0Ev.ptrauth.15 to i8*)], [11 x i8*] [i8* inttoptr (i64 -32 to i8*), i8* null, i8* inttoptr (i64 -32 to i8*), i8* inttoptr (i64 -32 to i8*), i8* inttoptr (i64 -32 to i8*), i8* bitcast (i8** @_ZTI2V0 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n24_N2V02m0Ev.ptrauth.16 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth.17 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.18 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2V0D1Ev.ptrauth.19 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2V0D0Ev.ptrauth.20 to i8*)] }

// CHECK: @_ZN2V02m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZN2V02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V0*)* @_ZN2V02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V0* (%class.V0*)* @_ZN2V0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZN2V0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V02m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZTv0_n24_N2V02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V0*)* @_ZTcv0_n32_h4_N2V02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.5 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V0* (%class.V0*)* @_ZTv0_n48_N2V0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZTv0_n48_N2V0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"

// CHECK: @_ZTC2D316_2V1 = unnamed_addr constant { [7 x i8*], [11 x i8*] } { [7 x i8*] [i8* inttoptr (i64 16 to i8*), i8* null, i8* bitcast (i8** @_ZTI2V1 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V12m0Ev.ptrauth.21 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V12m1Ev.ptrauth.22 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V1D1Ev.ptrauth.23 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2V1D0Ev.ptrauth.24 to i8*)], [11 x i8*] [i8* inttoptr (i64 -16 to i8*), i8* null, i8* inttoptr (i64 -16 to i8*), i8* inttoptr (i64 -16 to i8*), i8* inttoptr (i64 -16 to i8*), i8* bitcast (i8** @_ZTI2V1 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n24_N2V12m0Ev.ptrauth.25 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth.26 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN2B02m2Ev.ptrauth.27 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2V1D1Ev.ptrauth.28 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTv0_n48_N2V1D0Ev.ptrauth.29 to i8*)] }
// CHECK: @_ZN2V12m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZN2V12m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V1*)* @_ZN2V12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V1* (%class.V1*)* @_ZN2V1D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZN2V1D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V12m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZTv0_n24_N2V12m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V1*)* @_ZTcv0_n32_h4_N2V12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.6 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V1* (%class.V1*)* @_ZTv0_n48_N2V1D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZTv0_n48_N2V1D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2D32m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZN2D32m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2D32m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D3*)* @_ZN2D32m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2D3D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D3* (%class.D3*)* @_ZN2D3D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2D3D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZN2D3D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D32m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZThn16_N2D32m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 1, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D32m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D3*)* @_ZThn16_N2D32m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 1, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D3D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D3* (%class.D3*)* @_ZThn16_N2D3D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 1, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZThn16_N2D3D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZThn16_N2D3D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 1, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2D32m0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZTv0_n24_N2D32m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 2, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2D32m1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.D3*)* @_ZTcv0_n32_h4_N2D32m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 2, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.11 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 2, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2D3D1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.D3* (%class.D3*)* @_ZTv0_n48_N2D3D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 2, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2D3D0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.D3*)* @_ZTv0_n48_N2D3D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [7 x i8*], [11 x i8*] }, { [7 x i8*], [7 x i8*], [11 x i8*] }* @_ZTV2D3, i32 0, i32 2, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m0Ev.ptrauth.12 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZN2V02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 3) to i64), i64 44578 }, section "llvm.ptrauth"
// CHECK: @_ZN2V02m1Ev.ptrauth.13 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V0*)* @_ZN2V02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 4) to i64), i64 30766 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D1Ev.ptrauth.14 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V0* (%class.V0*)* @_ZN2V0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 5) to i64), i64 57279 }, section "llvm.ptrauth"
// CHECK: @_ZN2V0D0Ev.ptrauth.15 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZN2V0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 0, i32 6) to i64), i64 62452 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V02m0Ev.ptrauth.16 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZTv0_n24_N2V02m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V02m1Ev.ptrauth.17 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V0*)* @_ZTcv0_n32_h4_N2V02m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.18 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D1Ev.ptrauth.19 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V0* (%class.V0*)* @_ZTv0_n48_N2V0D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V0D0Ev.ptrauth.20 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V0*)* @_ZTv0_n48_N2V0D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D30_2V0, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m0Ev.ptrauth.21 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZN2V12m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 3) to i64), i64 49430 }, section "llvm.ptrauth"
// CHECK: @_ZN2V12m1Ev.ptrauth.22 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V1*)* @_ZN2V12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 4) to i64), i64 57119 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D1Ev.ptrauth.23 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V1* (%class.V1*)* @_ZN2V1D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 5) to i64), i64 60799 }, section "llvm.ptrauth"
// CHECK: @_ZN2V1D0Ev.ptrauth.24 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZN2V1D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 0, i32 6) to i64), i64 52565 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n24_N2V12m0Ev.ptrauth.25 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZTv0_n24_N2V12m0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 6) to i64), i64 53119 }, section "llvm.ptrauth"
// CHECK: @_ZTcv0_n32_h4_N2V12m1Ev.ptrauth.26 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%struct.S2* (%class.V1*)* @_ZTcv0_n32_h4_N2V12m1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 7) to i64), i64 15165 }, section "llvm.ptrauth"
// CHECK: @_ZN2B02m2Ev.ptrauth.27 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.B0*)* @_ZN2B02m2Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 8) to i64), i64 43073 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D1Ev.ptrauth.28 = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.V1* (%class.V1*)* @_ZTv0_n48_N2V1D1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 9) to i64), i64 25525 }, section "llvm.ptrauth"
// CHECK: @_ZTv0_n48_N2V1D0Ev.ptrauth.29 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%class.V1*)* @_ZTv0_n48_N2V1D0Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*], [11 x i8*] }, { [7 x i8*], [11 x i8*] }* @_ZTC2D316_2V1, i32 0, i32 1, i32 10) to i64), i64 21295 }, section "llvm.ptrauth"


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

// CHECK-LABEL: define %class.V1* @_ZN2V1D2Ev(
// CHECK: %[[T0:[0-9]+]] = load i8*, i8** %{{.*}}
// CHECK: %[[T1:[0-9]+]] = ptrtoint i8* %[[T0]] to i64
// CHECK: %[[T2:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T1]], i32 2, i64 0)
// CHECK: %[[T3:[0-9]+]] = inttoptr i64 %[[T2]] to i8*
// CHECK: %[[SLOT:[0-9]+]] = bitcast %class.V1* %{{.*}} to i32 (...)***
// CHECK: %[[T5:[0-9]+]] = bitcast i8* %[[T3]] to i32 (...)**
// CHECK: %[[T6:[0-9]+]] = ptrtoint i32 (...)** %[[T5]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 %[[T6]], i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T7]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGNED_VTADDR]], i32 (...)*** %[[SLOT]]

// CHECK-LABEL: define void @_Z8testB0m0P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.B0*)**, void (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.B0*)*, void (%class.B0*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%class.B0*)*, void (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m0(B0 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testB0m1P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load %struct.S1* (%class.B0*)**, %struct.S1* (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint %struct.S1* (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to %struct.S1* (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds %struct.S1* (%class.B0*)*, %struct.S1* (%class.B0*)** %[[T4]], i64 1
// CHECK: %[[T5:[0-9]+]] = load %struct.S1* (%class.B0*)*, %struct.S1* (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint %struct.S1* (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 15165)
// CHECK: call %struct.S1* %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m1(B0 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testB0m2P2B0(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.B0*)**, void (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.B0*)*, void (%class.B0*)** %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load void (%class.B0*)*, void (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testB0m2(B0 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z8testD0m0P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D0*)**, void (%class.D0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D0*)*, void (%class.D0*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%class.D0*)*, void (%class.D0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](%class.D0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m0(D0 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD0m1P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load %struct.S2* (%class.D0*)**, %struct.S2* (%class.D0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint %struct.S2* (%class.D0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to %struct.S2* (%class.D0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds %struct.S2* (%class.D0*)*, %struct.S2* (%class.D0*)** %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load %struct.S2* (%class.D0*)*, %struct.S2* (%class.D0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint %struct.S2* (%class.D0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 35045)
// CHECK: call %struct.S2* %[[T5]](%class.D0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m1(D0 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testD0m2P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.B0*)**, void (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.B0*)*, void (%class.B0*)** %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load void (%class.B0*)*, void (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m2(D0 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z8testD0m3P2D0(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D0*)**, void (%class.D0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D0*)*, void (%class.D0*)** %[[T4]], i64 6
// CHECK: %[[T5:[0-9]+]] = load void (%class.D0*)*, void (%class.D0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 10565)
// CHECK: call void %[[T5]](%class.D0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD0m3(D0 *a) {
  a->m3();
}


// CHECK-LABEL: define void @_Z8testD1m0P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D1*)**, void (%class.D1*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D1*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D1*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D1*)*, void (%class.D1*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%class.D1*)*, void (%class.D1*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D1*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](%class.D1* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m0(D1 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD1m1P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load %struct.S2* (%class.D1*)**, %struct.S2* (%class.D1*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint %struct.S2* (%class.D1*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to %struct.S2* (%class.D1*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds %struct.S2* (%class.D1*)*, %struct.S2* (%class.D1*)** %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load %struct.S2* (%class.D1*)*, %struct.S2* (%class.D1*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint %struct.S2* (%class.D1*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 52864)
// CHECK: call %struct.S2* %[[T5]](%class.D1* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m1(D1 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testD1m2P2D1(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.B0*)**, void (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.B0*)*, void (%class.B0*)** %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load void (%class.B0*)*, void (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD1m2(D1 *a) {
  a->m2();
}


// CHECK-LABEL: define void @_Z8testD2m0P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D2*)**, void (%class.D2*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D2*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D2*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D2*)*, void (%class.D2*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%class.D2*)*, void (%class.D2*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D2*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 53119)
// CHECK: call void %[[T5]](%class.D2* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m0(D2 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD2m1P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load %struct.S2* (%class.D2*)**, %struct.S2* (%class.D2*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint %struct.S2* (%class.D2*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to %struct.S2* (%class.D2*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds %struct.S2* (%class.D2*)*, %struct.S2* (%class.D2*)** %[[T4]], i64 5
// CHECK: %[[T5:[0-9]+]] = load %struct.S2* (%class.D2*)*, %struct.S2* (%class.D2*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint %struct.S2* (%class.D2*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 35045)
// CHECK: call %struct.S2* %[[T5]](%class.D2* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m1(D2 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z10testD2m2D0P2D2(
// CHECK: call void @_ZN2B02m2Ev(%class.B0* %{{.*}}){{$}}

void testD2m2D0(D2 *a) {
  a->D0::m2();
}

// CHECK-LABEL: define void @_Z10testD2m2D1P2D2(
// CHECK: call void @_ZN2B02m2Ev(%class.B0* %{{.*}}){{$}}

void testD2m2D1(D2 *a) {
  a->D1::m2();
}

// CHECK-LABEL: define void @_Z8testD2m3P2D2(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D2*)**, void (%class.D2*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D2*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D2*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D2*)*, void (%class.D2*)** %[[T4]], i64 6
// CHECK: %[[T5:[0-9]+]] = load void (%class.D2*)*, void (%class.D2*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D2*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 10565)
// CHECK: call void %[[T5]](%class.D2* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD2m3(D2 *a) {
  a->m3();
}

// CHECK-LABEL: define void @_Z8testD3m0P2D3(
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D3*)**, void (%class.D3*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.D3*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D3*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D3*)*, void (%class.D3*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%class.D3*)*, void (%class.D3*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D3*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 44578)
// CHECK: call void %[[T5]](%class.D3* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3m0(D3 *a) {
  a->m0();
}

// CHECK-LABEL: define void @_Z8testD3m1P2D3(
// CHECK: %[[VTABLE:[a-z]+]] = load %struct.S2* (%class.D3*)**, %struct.S2* (%class.D3*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint %struct.S2* (%class.D3*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to %struct.S2* (%class.D3*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds %struct.S2* (%class.D3*)*, %struct.S2* (%class.D3*)** %[[T4]], i64 1
// CHECK: %[[T5:[0-9]+]] = load %struct.S2* (%class.D3*)*, %struct.S2* (%class.D3*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint %struct.S2* (%class.D3*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 30766)
// CHECK: call %struct.S2* %[[T5]](%class.D3* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3m1(D3 *a) {
  a->m1();
}

// CHECK-LABEL: define void @_Z8testD3m2P2D3(
// CHECK: %[[VTABLE:[a-z0-9]+]] = load void (%class.B0*)**, void (%class.B0*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.B0*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.B0*)*, void (%class.B0*)** %[[T4]], i64 2
// CHECK: %[[T5:[0-9]+]] = load void (%class.B0*)*, void (%class.B0*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.B0*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 43073)
// CHECK: call void %[[T5]](%class.B0* %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3m2(D3 *a) {
  a->m2();
}

// CHECK-LABEL: define void @_Z17testD3Destructor0P2D3(
// CHECK: %[[T1:[0-9]+]] = bitcast %class.D3* %{{.*}} to void (%class.D3*)***
// CHECK: %[[VTABLE:[a-z]+]] = load void (%class.D3*)**, void (%class.D3*)*** %[[T1]]
// CHECK: %[[T2:[0-9]+]] = ptrtoint void (%class.D3*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%class.D3*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%class.D3*)*, void (%class.D3*)** %[[T4]], i64 3
// CHECK: %[[T5:[0-9]+]] = load void (%class.D3*)*, void (%class.D3*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%class.D3*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 62452)
// CHECK: call void %[[T5]](%class.D3* %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T7]]) ]

void testD3Destructor0(D3 *a) {
  delete a;
}

// CHECK-LABEL: define void @_Z17testD3Destructor1P2D3(
// CHECK: %[[T1:[0-9]+]] = bitcast %class.D3* %{{.*}} to i64**
// CHECK: %[[VTABLE0:[a-z0-9]+]] = load i64*, i64** %[[T1]]
// CHECK: %[[T2:[0-9]+]] = ptrtoint i64* %[[VTABLE0]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to i64*
// CHECK: %[[COMPLETE_OFFSET_PTR:.*]] = getelementptr inbounds i64, i64* %[[T4]], i64 -2
// CHECK: %[[T5:[0-9]+]] = load i64, i64* %[[COMPLETE_OFFSET_PTR]]
// CHECK: %[[T6:[0-9]+]] = bitcast %class.D3* %{{.*}} to i8*
// CHECK: %[[T7:[0-9]+]] = getelementptr inbounds i8, i8* %[[T6]], i64 %[[T5]]
// CHECK: %[[T8:[0-9]+]] = bitcast %class.D3* %{{.*}} to %class.D3* (%class.D3*)***
// CHECK: %[[VTABLE1:[a-z0-9]+]] = load %class.D3* (%class.D3*)**, %class.D3* (%class.D3*)*** %[[T8]]
// CHECK: %[[T9:[0-9]+]] = ptrtoint %class.D3* (%class.D3*)** %[[VTABLE1]] to i64
// CHECK: %[[T10:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T9]], i32 2, i64 0)
// CHECK: %[[T11:[0-9]+]] = inttoptr i64 %[[T10]] to %class.D3* (%class.D3*)**
// CHECK: %[[VFN:[a-z0-9]+]] = getelementptr inbounds %class.D3* (%class.D3*)*, %class.D3* (%class.D3*)** %[[T11]], i64 2
// CHECK: %[[T12:[0-9]+]] = load %class.D3* (%class.D3*)*, %class.D3* (%class.D3*)** %[[VFN]]
// CHECK: %[[T13:[0-9]+]] = ptrtoint %class.D3* (%class.D3*)** %[[VFN]] to i64
// CHECK: %[[T14:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T13]], i64 57279)
// CHECK: %call = call %class.D3* %[[T12]](%class.D3* %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T14]]) ]
// CHECK: call void @_ZdlPv(i8* %[[T7]])

void testD3Destructor1(D3 *a) {
  ::delete a;
}

// CHECK-LABEL: define void @_Z17testD3Destructor2P2D3(
// CHECK: %[[T1:.*]] = bitcast %class.D3* %{{.*}} to %class.D3* (%class.D3*)***
// CHECK: %[[VTABLE:.*]] = load %class.D3* (%class.D3*)**, %class.D3* (%class.D3*)*** %[[T1]]
// CHECK: %[[T2:.*]] = ptrtoint %class.D3* (%class.D3*)** %[[VTABLE]] to i64
// CHECK: %[[T3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T2]], i32 2, i64 0)
// CHECK: %[[T4:.*]] = inttoptr i64 %[[T3]] to %class.D3* (%class.D3*)**
// CHECK: %[[VFN:.*]] = getelementptr inbounds %class.D3* (%class.D3*)*, %class.D3* (%class.D3*)** %[[T4]], i64 2
// CHECK: %[[T5:.*]] = load %class.D3* (%class.D3*)*, %class.D3* (%class.D3*)** %[[VFN]]
// CHECK: %[[T6:.*]] = ptrtoint %class.D3* (%class.D3*)** %[[VFN]] to i64
// CHECK: %[[T7:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 57279)
// CHECK: %call = call %class.D3* %[[T5]](%class.D3* %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 %[[T7]]) ]

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

// CHECK-LABEL: define linkonce_odr %class.B0* @_ZN2B0C2Ev(
// CHECK: %[[THIS:[0-9]+]] = bitcast %class.B0* %{{.*}} to i32 (...)***
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i8** getelementptr inbounds ({ [7 x i8*] }, { [7 x i8*] }* @_ZTV2B0, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGNED_VTADDR]], i32 (...)*** %[[THIS]]

// CHECK-LABEL: define linkonce_odr %class.D0* @_ZN2D0C2Ev(
// CHECK: %[[THIS:[0-9]+]] = bitcast %class.D0* %{{.*}} to i32 (...)***
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV2D0, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGNED_VTADDR]], i32 (...)*** %[[THIS]]

// CHECK-LABEL: define linkonce_odr %class.D1* @_ZN2D1C2Ev(
// CHECK: %[[THIS:[0-9]+]] = bitcast %class.D1* %{{.*}} to i32 (...)***
// CHECK: %[[T0:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i8** getelementptr inbounds ({ [8 x i8*] }, { [8 x i8*] }* @_ZTV2D1, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[SIGNED_VTADDR:[0-9]+]] = inttoptr i64 %[[T0]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGNED_VTADDR]], i32 (...)*** %[[THIS]]

// CHECK-LABEL: define linkonce_odr %class.D2* @_ZN2D2C2Ev(
// CHECK: %[[SLOT0:[0-9+]]] = bitcast %class.D2* %[[THIS:[a-z0-9]+]] to i32 (...)***
// CHECK: %[[SIGN_VTADDR0:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, inrange i32 0, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[T1:[0-9]+]] = inttoptr i64 %[[SIGN_VTADDR0]] to i32 (...)**
// CHECK: store i32 (...)** %[[T1]], i32 (...)*** %[[SLOT0]]
// CHECK: %[[T2:[0-9]+]] = bitcast %class.D2* %[[THIS]] to i8*
// CHECK: %[[T3:[a-z0-9.]+]] = getelementptr inbounds i8, i8* %[[T2]], i64 16
// CHECK: %[[SLOT1:[0-9]+]] = bitcast i8* %[[T3]] to i32 (...)***
// CHECK: %[[SIGN_VTADDR1:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @_ZTV2D2, i32 0, inrange i32 1, i32 2) to i64), i32 2, i64 0)
// CHECK: %[[T5:[0-9]+]] = inttoptr i64 %[[SIGN_VTADDR1]] to i32 (...)**
// CHECK: store i32 (...)** %[[T5]], i32 (...)*** %[[SLOT1]]

// CHECK-LABEL: define linkonce_odr %class.V0* @_ZN2V0C2Ev(
// CHECK: %[[VTT:[a-z0-9]+]] = load i8**, i8*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = load i8*, i8** %[[VTT]]
// CHECK: %[[T1:[0-9]+]] = ptrtoint i8* %[[T0]] to i64
// CHECK: %[[T2:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T1]], i32 2, i64 0)
// CHECK: %[[T3:[0-9]+]] = inttoptr i64 %[[T2]] to i8*
// CHECK: %[[SLOT0:[0-9]+]] = bitcast %class.V0* %[[THIS:[a-z0-9]+]] to i32 (...)***
// CHECK: %[[T5:[0-9]+]] = bitcast i8* %[[T3]] to i32 (...)**
// CHECK: %[[VTADDR0:[0-9]+]] = ptrtoint i32 (...)** %[[T5]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 %[[VTADDR0]], i32 2, i64 0)
// CHECK: %[[SIGN_VTADDR0:[0-9]+]] = inttoptr i64 %[[T7]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGN_VTADDR0]], i32 (...)*** %[[SLOT0]]
// CHECK: %[[T9:[0-9]+]] = getelementptr inbounds i8*, i8** %[[VTT]], i64 1
// CHECK: %[[T10:[0-9]+]] = load i8*, i8** %[[T9]]
// CHECK: %[[T11:[0-9]+]] = ptrtoint i8* %[[T10]] to i64
// CHECK: %[[T12:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T11]], i32 2, i64 0)
// CHECK: %[[T13:[0-9]+]] = inttoptr i64 %[[T12]] to i8*
// CHECK: %[[T14:[0-9]+]] = bitcast %class.V0* %[[THIS]] to i8**
// CHECK: %[[VTABLE:[a-z]+]] = load i8*, i8** %[[T14]]
// CHECK: %[[T15:[0-9]+]] = ptrtoint i8* %[[VTABLE]] to i64
// CHECK: %[[T16:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T15]], i32 2, i64 0)
// CHECK: %[[T17:[0-9]+]] = inttoptr i64 %[[T16]] to i8*
// CHECK: %[[VBASE_OFFSET_PTR:[a-z.]+]] = getelementptr i8, i8* %[[T17]], i64 -24
// CHECK: %[[T18:[0-9]+]] = bitcast i8* %[[VBASE_OFFSET_PTR]] to i64*
// CHECK: %[[VBASE_OFFSET:[a-z.]+]] = load i64, i64* %[[T18]]
// CHECK: %[[T19:[0-9]+]] = bitcast %class.V0* %[[THIS]] to i8*
// CHECK: %[[T20:[a-z.]+]] = getelementptr inbounds i8, i8* %[[T19]], i64 %[[VBASE_OFFSET]]
// CHECK: %[[SLOT1:[0-9]+]] = bitcast i8* %[[T20]] to i32 (...)***
// CHECK: %[[T21:[0-9]+]] = bitcast i8* %[[T13]] to i32 (...)**
// CHECK: %[[VTADDR1:[0-9]+]] = ptrtoint i32 (...)** %[[T21]] to i64
// CHECK: %[[T23:[0-9]+]] = call i64 @llvm.ptrauth.sign.i64(i64 %[[VTADDR1]], i32 2, i64 0)
// CHECK: %[[SIGN_VTADDR1:[0-9]+]] = inttoptr i64 %[[T23]] to i32 (...)**
// CHECK: store i32 (...)** %[[SIGN_VTADDR1]], i32 (...)*** %[[SLOT1]]
