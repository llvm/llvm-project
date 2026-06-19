// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-pauthtest -fptrauth-calls -disable-llvm-passes \
// RUN:     -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-pauthtest -fptrauth-calls -fptrauth-vtt-vtable-pointer-discrimination \
// RUN:     -disable-llvm-passes -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-pauthtest -fptrauth-calls -fptrauth-vtt-vtable-pointer-discrimination \
// RUN:     -fptrauth-vtable-pointer-address-discrimination -disable-llvm-passes -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,ADDRESS %s
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-pauthtest -fptrauth-calls -fptrauth-vtt-vtable-pointer-discrimination \
// RUN:     -fptrauth-vtable-pointer-type-discrimination -disable-llvm-passes -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,TYPE %s
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple aarch64-linux-pauthtest -fptrauth-calls -fptrauth-vtt-vtable-pointer-discrimination \
// RUN:     -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination       \
// RUN:     -disable-llvm-passes -emit-llvm -O0 -o - | FileCheck --check-prefixes=CHECK,ADDRESSTYPE %s

// CHECK:      @_ZTT1D = linkonce_odr unnamed_addr constant [4 x ptr] [
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2), 
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2), 
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2), 
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2)]

// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 0, ptr @_ZTT1D), 
// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 0, ptr getelementptr (ptr, ptr @_ZTT1D, i32 1)), 
// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 0, ptr getelementptr (ptr, ptr @_ZTT1D, i32 2)), 
// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 0, ptr getelementptr (ptr, ptr @_ZTT1D, i32 3))]

// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 46475), 
// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 62866), 
// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 62866), 
// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 46475)]

// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 46475, ptr @_ZTT1D),
// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 62866, ptr getelementptr (ptr, ptr @_ZTT1D, i32 1)),
// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTC1D0_1A, i32 0, i32 0, i32 4), i32 2, i64 62866, ptr getelementptr (ptr, ptr @_ZTT1D, i32 2)),
// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 4), i32 2, i64 46475, ptr getelementptr (ptr, ptr @_ZTT1D, i32 3))]

// CHECK:      @_ZTT1A = linkonce_odr unnamed_addr constant [2 x ptr] [
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2), 
// DEFAULT-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2)]

// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 0, ptr @_ZTT1A), 
// ADDRESS-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 0, ptr getelementptr (ptr, ptr @_ZTT1A, i32 1))]

// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 62866),
// TYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 62866)]

// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 62866, ptr @_ZTT1A),
// ADDRESSTYPE-SAME: ptr ptrauth (ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 4), i32 2, i64 62866, ptr getelementptr (ptr, ptr @_ZTT1A, i32 1))]

// CHECK-LABEL: @_ZN1AC2Ev
// CHECK:       %vtt.addr = alloca ptr, align 8
// CHECK:       store ptr %vtt, ptr %vtt.addr, align 8
// CHECK:       [[VTT:%.*]] = load ptr, ptr %vtt.addr, align 8
// CHECK:       [[VTABLE:%.*]] = load ptr, ptr [[VTT]], align 8
// ADDRESS:     [[VTABLE_ADDR:%.*]] = ptrtoint ptr [[VTT]] to i64
// ADDRESSTYPE: [[VTABLE_ADDR:%.*]] = ptrtoint ptr [[VTT]] to i64
// ADDRESSTYPE: [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTABLE_ADDR]], i64 62866)
// CHECK:       [[VTABLEi64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// DEFAULT:     call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 0)
// ADDRESS:     call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 [[VTABLE_ADDR]])
// TYPE:        call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 62866)
// ADDRESSTYPE: call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 [[DISC]])

// CHECK:       [[VTTOFFSET:%.*]]  = getelementptr inbounds ptr, ptr [[VTT]], i64 1
// CHECK:       [[VTABLE2:%.*]] = load ptr, ptr [[VTTOFFSET]], align 8
// ADDRESS:     [[VTABLE2_ADDR:%.*]] = ptrtoint ptr [[VTTOFFSET]] to i64
// ADDRESSTYPE: [[VTABLE2_ADDR:%.*]] = ptrtoint ptr [[VTTOFFSET]] to i64
// ADDRESSTYPE: [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTABLE2_ADDR]], i64 62866)
// CHECK:       [[VTABLE2i64:%.*]] = ptrtoint ptr [[VTABLE2]] to i64
// DEFAULT:     call i64 @llvm.ptrauth.auth(i64 [[VTABLE2i64]], i32 2, i64 0)
// ADDRESS:     call i64 @llvm.ptrauth.auth(i64 [[VTABLE2i64]], i32 2, i64 [[VTABLE2_ADDR]])
// TYPE:        call i64 @llvm.ptrauth.auth(i64 [[VTABLE2i64]], i32 2, i64 62866)
// ADDRESSTYPE: call i64 @llvm.ptrauth.auth(i64 [[VTABLE2i64]], i32 2, i64 [[DISC]])

// CHECK-LABEL: @_ZN1DC2Ev
// CHECK:   %vtt.addr = alloca ptr, align 8
// CHECK:   store ptr %vtt, ptr %vtt.addr, align 8
// CHECK:   [[VTT:%.*]] = load ptr, ptr %vtt.addr, align 8
// CHECK:   [[VTTOFFSET:%.*]] = getelementptr inbounds ptr, ptr [[VTT]], i64 1
// CHECK:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1, ptr noundef [[VTTOFFSET]])
// CHECK:   [[VTABLE:%.*]] = load ptr, ptr [[VTT]], align 8
// ADDRESS: [[VTABLE_ADDR:%.*]] = ptrtoint ptr [[VTT]] to i64
// CHECK:   [[VTABLEi64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// DEFAULT: call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 0)
// ADDRESS: call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 [[VTABLE_ADDR]])
// TYPE:    call i64 @llvm.ptrauth.auth(i64 [[VTABLEi64]], i32 2, i64 46475)

struct V {
    virtual void f();
};

struct A : virtual V {
    A();
};

struct D : A {
    D();
};

A::A() {}
D::D() {}
