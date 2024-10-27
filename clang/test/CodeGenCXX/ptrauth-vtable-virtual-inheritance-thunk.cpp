// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple arm64-apple-ios   -disable-llvm-passes -fptrauth-intrinsics -fptrauth-calls \
// RUN:   -fptrauth-vtable-pointer-type-discrimination -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,DARWIN %s
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-gnu -disable-llvm-passes -fptrauth-intrinsics -fptrauth-calls \
// RUN:   -fptrauth-vtable-pointer-type-discrimination -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,ELF %s

// The actual vtable construction

// CHECK: @_ZTV1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD1Ev, i32 0, i64 31214, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD0Ev, i32 0, i64 8507, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 4))],
// CHECK-SAME: [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTT1C = unnamed_addr constant [2 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 6), i32 2)], align 8

// CHECK: @_ZTV1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD1Ev, i32 0, i64 59423, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD0Ev, i32 0, i64 25900, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1gEv, i32 0, i64 59070, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1hEz, i32 0, i64 65100, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 6))],
// CHECK-SAME: [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTT1D = unnamed_addr constant [2 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 6), i32 2)], align 8

// CHECK: @_ZTV1F = unnamed_addr constant { [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] } { [6 x ptr] [ptr inttoptr (i64 32 to ptr), ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1F,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1FD1Ev, i32 0, i64 31214, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1FD0Ev, i32 0, i64 8507, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 5))], [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1F,
// CHECK-SAME: ptr ptrauth (ptr @_ZThn8_N1FD1Ev, i32 0, i64 59423, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZThn8_N1FD0Ev, i32 0, i64 25900, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1gEv, i32 0, i64 59070, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1hEz, i32 0, i64 65100, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 6))], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1F,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1FD1EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 2043, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 63674, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 10))], [11 x ptr] [ptr inttoptr (i64 -32 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -32 to ptr), ptr @_ZTI1F,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1fEv, i32 0, i64 28408, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1gEv, i32 0, i64 22926, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1hEz, i32 0, i64 9832, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1FD1Ev, i32 0, i64 5817, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1FD0Ev, i32 0, i64 26464, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 10))] }, align 8

// CHECK: @_ZTT1F = unnamed_addr constant [8 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 0, i32 4), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 2, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 32) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 1, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1F, i32 0, i32 3, i32 6), i32 2)], align 8

// CHECK: @_ZTV1G = unnamed_addr constant { [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] } { [6 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1G,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1GD1Ev, i32 0, i64 31214, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1GD0Ev, i32 0, i64 8507, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 5))], [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1G,
// CHECK-SAME: ptr ptrauth (ptr @_ZThn8_N1GD1Ev, i32 0, i64 59423, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZThn8_N1GD0Ev, i32 0, i64 25900, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1gEv, i32 0, i64 59070, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1hEz, i32 0, i64 65100, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 6))], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1G,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1fEv, i32 0, i64 28408, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1gEv, i32 0, i64 22926, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1hEz, i32 0, i64 9832, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1GD1Ev, i32 0, i64 5817, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1GD0Ev, i32 0, i64 26464, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 10))], [11 x ptr] [ptr inttoptr (i64 -24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1G,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1GD1EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 2043, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E, i32 0, i64 63674, ptr getelementptr inbounds ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 10))] }, align 8

// CHECK: @_ZTT1G = unnamed_addr constant [8 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 0, i32 4), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 32) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 2, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-48, 40) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 3, i32 6), i32 2),
// CHECK-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-24, 32) ({ [6 x ptr], [7 x ptr], [11 x ptr], [11 x ptr] }, ptr @_ZTV1G, i32 0, i32 1, i32 3), i32 2)], align 8

// CHECK: @_ZTV1A = unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1A,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1AD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1AD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 6))] }, align 8

// CHECK: @_ZTI1A = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2), ptr @_ZTS1A }, align 8

// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]

// CHECK: @_ZTS1A = constant [3 x i8] c"1A\00", align 1

// CHECK: @_ZTI1C = constant { ptr, ptr, i32, i32, ptr, i64 } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), i32 2), ptr @_ZTS1C, i32 0, i32 1, ptr @_ZTI1B, i64 -6141 }, align 8

// CHECK: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]

// CHECK: @_ZTS1C = constant [3 x i8] c"1C\00", align 1

// DARWIN: @_ZTI1B = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), i32 2), ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTS1B to i64), i64 -9223372036854775808) to ptr), ptr @_ZTI1A }, align 8
// ELF:    @_ZTI1B = linkonce_odr constant { ptr, ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), i32 2), ptr @_ZTS1B, ptr @_ZTI1A }, comdat, align 8

// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]

// DARWIN: @_ZTS1B = linkonce_odr hidden constant [3 x i8] c"1B\00", align 1
// ELF:    @_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00", comdat, align 1

// CHECK: @_ZTI1D = constant { ptr, ptr, i32, i32, ptr, i64 } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), i32 2), ptr @_ZTS1D, i32 0, i32 1, ptr @_ZTI1B, i64 -6141 }, align 8

// CHECK: @_ZTS1D = constant [3 x i8] c"1D\00", align 1

// CHECK: @_ZTV1E = unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1E,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1fEv, i32 0, i64 28408, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 2)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1gEv, i32 0, i64 22926, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1E1hEz, i32 0, i64 9832, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1ED1Ev, i32 0, i64 5817, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1ED0Ev, i32 0, i64 26464, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1E, i32 0, i32 0, i32 6))] }, align 8

// CHECK: @_ZTI1E = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2), ptr @_ZTS1E }, align 8

// CHECK: @_ZTS1E = constant [3 x i8] c"1E\00", align 1

// CHECK: @_ZTC1F0_1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD1Ev, i32 0, i64 31214, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD0Ev, i32 0, i64 8507, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 0, i32 4))], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1F0_1C, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTC1F8_1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD1Ev, i32 0, i64 59423, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD0Ev, i32 0, i64 25900, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1gEv, i32 0, i64 59070, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1hEz, i32 0, i64 65100, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 0, i32 6))], [11 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1F8_1D, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTI1F = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64 } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), i32 2), ptr @_ZTS1F, i32 3, i32 3, ptr @_ZTI1C, i64 2, ptr @_ZTI1D, i64 2050, ptr @_ZTI1E, i64 -8189 }, align 8

// CHECK: @_ZTS1F = constant [3 x i8] c"1F\00", align 1

// CHECK: @_ZTC1G0_1C = unnamed_addr constant { [5 x ptr], [11 x ptr] } { [5 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD1Ev, i32 0, i64 31214, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1CD0Ev, i32 0, i64 8507, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 0, i32 4))], [11 x ptr] [ptr inttoptr (i64 -24 to ptr), ptr null, ptr null, ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1C,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1CD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [5 x ptr], [11 x ptr] }, ptr @_ZTC1G0_1C, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTC1G8_1D = unnamed_addr constant { [7 x ptr], [11 x ptr] } { [7 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD1Ev, i32 0, i64 59423, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1DD0Ev, i32 0, i64 25900, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1gEv, i32 0, i64 59070, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 5)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1D1hEz, i32 0, i64 65100, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 0, i32 6))], [11 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 6)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n32_N1D1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 7)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n40_N1D1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 8)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 9)),
// CHECK-SAME: ptr ptrauth (ptr @_ZTv0_n48_N1DD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr], [11 x ptr] }, ptr @_ZTC1G8_1D, i32 0, i32 1, i32 10))] }, align 8

// CHECK: @_ZTI1G = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64 } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), i32 2), ptr @_ZTS1G, i32 3, i32 3, ptr @_ZTI1E, i64 -8189, ptr @_ZTI1C, i64 2, ptr @_ZTI1D, i64 2050 }, align 8

// CHECK: @_ZTS1G = constant [3 x i8] c"1G\00", align 1

// CHECK: @_ZTV1B = linkonce_odr unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr @_ZTI1B,
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1fEv, i32 0, i64 55636, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1gEv, i32 0, i64 19402, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1A1hEz, i32 0, i64 31735, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 4)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN1BD1Ev, i32 0, i64 2043, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 5)),
// DARWIN-SAME: ptr ptrauth (ptr @_ZN1BD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 6))] }, align 8
// ELF-SAME:    ptr ptrauth (ptr @_ZN1BD0Ev, i32 0, i64 63674, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 6))] }, comdat, align 8

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
// DARWIN: ptr @_ZTv0_n48_N1CD1Ev(ptr noundef %this)
// ELF:    void @_ZTv0_n48_N1CD1Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1CD0Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// DARWIN: ptr @_ZTv0_n48_N1DD1Ev(ptr noundef %this)
// ELF:    void @_ZTv0_n48_N1DD1Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1DD0Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1FD0EvU11__vtptrauthILj0Lb0Lj62866E(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)

// CHECK: void @_ZTv0_n48_N1FD0Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 12810)

// CHECK: void @_ZTv0_n48_N1GD0Ev(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 12810)

// CHECK: void @_ZTv0_n48_N1GD0EvU11__vtptrauthILj0Lb0Lj62866E(ptr noundef %this)
// CHECK: [[TEMP:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[TEMP:%.*]], i32 2, i64 62866)
