// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple arm64-apple-ios -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -emit-llvm -no-enable-noundef-analysis -O0 -disable-llvm-passes -o - | FileCheck --check-prefix=CHECK %s

// The actual vtable construction
// CHECK: @_ZTV1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1C, ptr @_ZN1CD1Ev.ptrauth, ptr @_ZN1CD0Ev.ptrauth], [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1C, ptr @_ZN1A1fEv.ptrauth.2, ptr @_ZN1A1gEv.ptrauth.3, ptr @_ZN1A1hEz.ptrauth.4, ptr @_ZTv0_n48_N1CD1Ev.ptrauth, ptr @_ZTv0_n48_N1CD0Ev.ptrauth] }, align 8
// CHECK: @_ZTT1C = unnamed_addr constant [2 x ptr] [ptr @_ZTV1C.ptrauth, ptr @_ZTV1C.ptrauth.1], align 8
// CHECK: @_ZTV1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1D, ptr @_ZN1DD1Ev.ptrauth, ptr @_ZN1DD0Ev.ptrauth, ptr @_ZN1D1gEv.ptrauth, ptr @_ZN1D1hEz.ptrauth], [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1D, ptr @_ZN1A1fEv.ptrauth.6, ptr @_ZTv0_n32_N1D1gEv.ptrauth, ptr @_ZTv0_n40_N1D1hEz.ptrauth, ptr @_ZTv0_n48_N1DD1Ev.ptrauth, ptr @_ZTv0_n48_N1DD0Ev.ptrauth] }, align 8
// CHECK: @_ZTT1D = unnamed_addr constant [2 x ptr] [ptr @_ZTV1D.ptrauth, ptr @_ZTV1D.ptrauth.5], align 8
// CHECK: @_ZTV1F = unnamed_addr constant { [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] } { [6 x ptr] [ptr inttoptr (i64 32 to ptr), ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1F, ptr @_ZN1FD1Ev.ptrauth, ptr @_ZN1FD0Ev.ptrauth], [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1F, ptr @_ZThn8_N1FD1Ev.ptrauth, ptr @_ZThn8_N1FD0Ev.ptrauth, ptr @_ZN1D1gEv.ptrauth.28, ptr @_ZN1D1hEz.ptrauth.29], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1F, ptr @_ZN1A1fEv.ptrauth.30, ptr @_ZTv0_n32_N1D1gEv.ptrauth.31, ptr @_ZTv0_n40_N1D1hEz.ptrauth.32, ptr @_ZTv0_n48_N1FD1EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth, ptr @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth], [11 x ptr] [ptr inttoptr (i64 -32 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -32 to ptr), ptr @_ZTI1F, ptr @_ZN1E1fEv.ptrauth.33, ptr @_ZN1E1gEv.ptrauth.34, ptr @_ZN1E1hEz.ptrauth.35, ptr @_ZTv0_n48_N1FD1Ev.ptrauth, ptr @_ZTv0_n48_N1FD0Ev.ptrauth] }, align 8
// CHECK: @_ZTT1F = unnamed_addr constant [8 x ptr] [ptr @_ZTV1F.ptrauth, ptr @_ZTC1F0_1C.ptrauth, ptr @_ZTC1F0_1C.ptrauth.23, ptr @_ZTC1F8_1D.ptrauth, ptr @_ZTC1F8_1D.ptrauth.24, ptr @_ZTV1F.ptrauth.25, ptr @_ZTV1F.ptrauth.26, ptr @_ZTV1F.ptrauth.27], align 8
// CHECK: @_ZTV1G = unnamed_addr constant { [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] } { [6 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1G, ptr @_ZN1GD1Ev.ptrauth, ptr @_ZN1GD0Ev.ptrauth], [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1G, ptr @_ZThn8_N1GD1Ev.ptrauth, ptr @_ZThn8_N1GD0Ev.ptrauth, ptr @_ZN1D1gEv.ptrauth.73, ptr @_ZN1D1hEz.ptrauth.74], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1G, ptr @_ZN1E1fEv.ptrauth.75, ptr @_ZN1E1gEv.ptrauth.76, ptr @_ZN1E1hEz.ptrauth.77, ptr @_ZTv0_n48_N1GD1Ev.ptrauth, ptr @_ZTv0_n48_N1GD0Ev.ptrauth], [11 x ptr] [ptr inttoptr (i64 -24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1G, ptr @_ZN1A1fEv.ptrauth.78, ptr @_ZTv0_n32_N1D1gEv.ptrauth.79, ptr @_ZTv0_n40_N1D1hEz.ptrauth.80, ptr @_ZTv0_n48_N1GD1EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth, ptr @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth] }, align 8
// CHECK: @_ZTT1G = unnamed_addr constant [8 x ptr] [ptr @_ZTV1G.ptrauth, ptr @_ZTC1G0_1C.ptrauth, ptr @_ZTC1G0_1C.ptrauth.68, ptr @_ZTC1G8_1D.ptrauth, ptr @_ZTC1G8_1D.ptrauth.69, ptr @_ZTV1G.ptrauth.70, ptr @_ZTV1G.ptrauth.71, ptr @_ZTV1G.ptrauth.72], align 8
// CHECK: @_ZTV1A = unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv.ptrauth, ptr @_ZN1A1gEv.ptrauth, ptr @_ZN1A1hEz.ptrauth, ptr @_ZN1AD1Ev.ptrauth, ptr @_ZN1AD0Ev.ptrauth] }, align 8
// CHECK: @_ZTI1A = constant { ptr, ptr } { ptr @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth, ptr @_ZTS1A }, align 8
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTS1A = constant [3 x i8] c"1A\00", align 1
// CHECK: @_ZN1A1fEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 3) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1AD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1AD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 5) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1AD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1AD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 6) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1C.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"
// CHECK: @_ZTV1C.ptrauth.1 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"
// CHECK: @_ZTI1C = constant { ptr, ptr, i32, i32, ptr, i64 } { ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE.ptrauth, ptr @_ZTS1C, i32 0, i32 1, ptr @_ZTI1B, i64 -6141 }, align 8
// CHECK: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTVN10__cxxabiv121__vmi_class_type_infoE.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTS1C = constant [3 x i8] c"1C\00", align 1
// CHECK: @_ZTI1B = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr @_ZTVN10__cxxabiv120__si_class_type_infoE.ptrauth, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTS1B to i64), i64 -9223372036854775808) to ptr), ptr @_ZTI1A }, align 8
// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTS1B = linkonce_odr hidden constant [3 x i8] c"1B\00", align 1
// CHECK: @_ZN1CD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.2 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.3 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.4 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1D.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1D.ptrauth.5 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTI1D = constant { ptr, ptr, i32, i32, ptr, i64 } { ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE.ptrauth, ptr @_ZTS1D, i32 0, i32 1, ptr @_ZTI1B, i64 -6141 }, align 8
// CHECK: @_ZTS1D = constant [3 x i8] c"1D\00", align 1
// CHECK: @_ZN1DD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.6 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1E = unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1E, ptr @_ZN1E1fEv.ptrauth, ptr @_ZN1E1gEv.ptrauth, ptr @_ZN1E1hEz.ptrauth, ptr @_ZN1ED1Ev.ptrauth, ptr @_ZN1ED0Ev.ptrauth] }, align 8
// CHECK: @_ZTI1E = constant { ptr, ptr } { ptr @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth, ptr @_ZTS1E }, align 8
// CHECK: @_ZTS1E = constant [3 x i8] c"1E\00", align 1
// CHECK: @_ZN1E1fEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 2) to i64), i64 28408 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1gEv.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 3) to i64), i64 22926 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1hEz.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 4) to i64), i64 9832 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1ED1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1ED1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 5) to i64), i64 5817 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1ED0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1ED0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 6) to i64), i64 26464 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F0_1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1C, ptr @_ZN1CD1Ev.ptrauth.97, ptr @_ZN1CD0Ev.ptrauth.98], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1C, ptr @_ZN1A1fEv.ptrauth.99, ptr @_ZN1A1gEv.ptrauth.100, ptr @_ZN1A1hEz.ptrauth.101, ptr @_ZTv0_n48_N1CD1Ev.ptrauth.102, ptr @_ZTv0_n48_N1CD0Ev.ptrauth.103] }, align 8
// CHECK: @_ZN1CD1Ev.ptrauth.7 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.8 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.9 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.10 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.11 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.12 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.13 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F8_1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1D, ptr @_ZN1DD1Ev.ptrauth.104, ptr @_ZN1DD0Ev.ptrauth.105, ptr @_ZN1D1gEv.ptrauth.106, ptr @_ZN1D1hEz.ptrauth.107], [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1D, ptr @_ZN1A1fEv.ptrauth.108, ptr @_ZTv0_n32_N1D1gEv.ptrauth.109, ptr @_ZTv0_n40_N1D1hEz.ptrauth.110, ptr @_ZTv0_n48_N1DD1Ev.ptrauth.111, ptr @_ZTv0_n48_N1DD0Ev.ptrauth.112] }, align 8
// CHECK: @_ZN1DD1Ev.ptrauth.14 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.15 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.16 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.17 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.18 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.19 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.20 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.21 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.22 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1F.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 4), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F0_1C.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F0_1C.ptrauth.23 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F8_1D.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1F8_1D.ptrauth.24 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1F.ptrauth.25 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1F.ptrauth.26 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 32) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1F.ptrauth.27 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTI1F = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64 } { ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE.ptrauth, ptr @_ZTS1F, i32 3, i32 3, ptr @_ZTI1C, i64 2, ptr @_ZTI1D, i64 2050, ptr @_ZTI1E, i64 -8189 }, align 8
// CHECK: @_ZTS1F = constant [3 x i8] c"1F\00", align 1
// CHECK: @_ZN1FD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1FD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 4) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1FD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1FD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 5) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZThn8_N1FD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn8_N1FD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZThn8_N1FD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn8_N1FD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.28 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.29 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.30 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.31 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.32 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1FD1EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1FD1EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1fEv.ptrauth.33 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 6) to i64), i64 28408 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1gEv.ptrauth.34 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 7) to i64), i64 22926 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1hEz.ptrauth.35 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 8) to i64), i64 9832 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1FD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1FD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 9) to i64), i64 5817 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1FD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1FD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 10) to i64), i64 26464 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD1Ev.ptrauth.36 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.37 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.38 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.39 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.40 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.41 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.42 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD1Ev.ptrauth.43 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.44 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.45 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.46 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.47 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.48 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.49 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.50 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.51 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G0_1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1C, ptr @_ZN1CD1Ev.ptrauth.113, ptr @_ZN1CD0Ev.ptrauth.114], [11 x ptr] [ptr inttoptr (i64 -24 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1C, ptr @_ZN1A1fEv.ptrauth.115, ptr @_ZN1A1gEv.ptrauth.116, ptr @_ZN1A1hEz.ptrauth.117, ptr @_ZTv0_n48_N1CD1Ev.ptrauth.118, ptr @_ZTv0_n48_N1CD0Ev.ptrauth.119] }, align 8
// CHECK: @_ZN1CD1Ev.ptrauth.52 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.53 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.54 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.55 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.56 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.57 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.58 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G8_1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1D, ptr @_ZN1DD1Ev.ptrauth.120, ptr @_ZN1DD0Ev.ptrauth.121, ptr @_ZN1D1gEv.ptrauth.122, ptr @_ZN1D1hEz.ptrauth.123], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D, ptr @_ZN1A1fEv.ptrauth.124, ptr @_ZTv0_n32_N1D1gEv.ptrauth.125, ptr @_ZTv0_n40_N1D1hEz.ptrauth.126, ptr @_ZTv0_n48_N1DD1Ev.ptrauth.127, ptr @_ZTv0_n48_N1DD0Ev.ptrauth.128] }, align 8
// CHECK: @_ZN1DD1Ev.ptrauth.59 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.60 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.61 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.62 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.63 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.64 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.65 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.66 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.67 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1G.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 4), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G0_1C.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G0_1C.ptrauth.68 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G8_1D.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTC1G8_1D.ptrauth.69 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1G.ptrauth.70 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1G.ptrauth.71 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 6), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1G.ptrauth.72 = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds inrange(-24, 32) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 3), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTI1G = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64 } { ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE.ptrauth, ptr @_ZTS1G, i32 3, i32 3, ptr @_ZTI1E, i64 -8189, ptr @_ZTI1C, i64 2, ptr @_ZTI1D, i64 2050 }, align 8
// CHECK: @_ZTS1G = constant [3 x i8] c"1G\00", align 1
// CHECK: @_ZN1GD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1GD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 4) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1GD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1GD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 5) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZThn8_N1GD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn8_N1GD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZThn8_N1GD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZThn8_N1GD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.73 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.74 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1fEv.ptrauth.75 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 6) to i64), i64 28408 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1gEv.ptrauth.76 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 7) to i64), i64 22926 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1E1hEz.ptrauth.77 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1E1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 8) to i64), i64 9832 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1GD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1GD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 9) to i64), i64 5817 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1GD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1GD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 10) to i64), i64 26464 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.78 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.79 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.80 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1GD1EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1GD1EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD1Ev.ptrauth.81 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.82 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.83 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.84 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.85 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.86 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.87 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD1Ev.ptrauth.88 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.89 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.90 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.91 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.92 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.93 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.94 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.95 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.96 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD1Ev.ptrauth.97 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.98 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.99 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.100 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.101 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.102 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.103 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD1Ev.ptrauth.104 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.105 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.106 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.107 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.108 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.109 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.110 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.111 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.112 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD1Ev.ptrauth.113 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3) to i64), i64 31214 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1CD0Ev.ptrauth.114 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 4) to i64), i64 8507 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.115 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.116 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.117 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD1Ev.ptrauth.118 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1CD0Ev.ptrauth.119 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD1Ev.ptrauth.120 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3) to i64), i64 59423 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1DD0Ev.ptrauth.121 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 4) to i64), i64 25900 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1gEv.ptrauth.122 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 5) to i64), i64 59070 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1D1hEz.ptrauth.123 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 6) to i64), i64 65100 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1fEv.ptrauth.124 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n32_N1D1gEv.ptrauth.125 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 7) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n40_N1D1hEz.ptrauth.126 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 8) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD1Ev.ptrauth.127 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 9) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTv0_n48_N1DD0Ev.ptrauth.128 = private constant { ptr, i32, i64, i64 } { ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 10) to i64), i64 63674 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTV1B = linkonce_odr unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1A1fEv.ptrauth.129, ptr @_ZN1A1gEv.ptrauth.130, ptr @_ZN1A1hEz.ptrauth.131, ptr @_ZN1BD1Ev.ptrauth, ptr @_ZN1BD0Ev.ptrauth] }, align 8
// CHECK: @_ZN1A1fEv.ptrauth.129 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1fEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64), i64 55636 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1gEv.ptrauth.130 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1gEv, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 3) to i64), i64 19402 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1A1hEz.ptrauth.131 = private constant { ptr, i32, i64, i64 } { ptr @_ZN1A1hEz, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 4) to i64), i64 31735 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1BD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1BD1Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 5) to i64), i64 2043 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN1BD0Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN1BD0Ev, i32 0, i64 ptrtoint (ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 6) to i64), i64 63674 }, section "llvm.ptrauth", align 8


extern "C" int printf(const char *format, ...);

class A {
public:
  A() {}
  virtual int f();
  virtual int g();
  virtual int h(...);
  virtual ~A() {}

public:
  bool necessary_field;
};

class B : public A {
public:
  B() : A() {}
  virtual ~B() {}
};

class C : public virtual B {
public:
  C() : B() {}
  ~C();
};

class D : public virtual B {
public:
  D() : B() {}
  ~D();
  virtual int g();
  virtual int h(...);
};

class E {
public:
  virtual int f();
  virtual int g();
  virtual int h(...);
  virtual ~E(){};
};

class F : public C, public D, public virtual E {
  ~F();
};

class G : public virtual E, public C, public D {
  ~G();
};

C::~C() {}
D::~D() {}
F::~F() {}
G::~G() {}
int E::f() { return 1; }
int A::f() { return 0; }
int E::g() { return 1; }
int A::g() { return 0; }
int D::g() { return 0; }

int E::h(...) { return 1; }
int A::h(...) { return 0; }
int D::h(...) { return 0; }

int main() {
  A *ans = new C();
  delete ans;

  B *b = new D();
  b->f();
  b->g();
  b->h(1,2,3);
  b = new C();
  b->f();
  b->h(1,2,3);
  b = new C();
  b->f();
  b->h(1,2,3);
  b = new F();
  b->f();
  b->g();
  b->h(1,2,3);

  ans = new B();
  delete ans;

  ans = new F();
  ans->f();
  ans->g();
  ans->h(1,2,3);
  delete ans;

  E *e = new F();
  e->f();
  e->g();
  e->h(1,2,3);
  delete e;
  e = new G();
  e->f();
  e->g();
  e->h(1,2,3);
  delete e;
}

// And check the thunks
// CHECK: ptr @_ZTv0_n48_N1CD1Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1CD0Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: ptr @_ZTv0_n48_N1DD1Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1DD0Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1FD0Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 12810)

// CHECK: void @_ZTv0_n48_N1GD0Ev(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 12810)

// CHECK: void @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E(ptr %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)
