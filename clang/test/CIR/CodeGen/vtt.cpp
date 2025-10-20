// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR-NO-RTTI,CIR-COMMON --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM-NO-RTTI,LLVM-COMMON --input-file=%t-cir.ll  %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG-NO-RTTI,OGCG-COMMON --input-file=%t.ll  %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR-RTTI,CIR-COMMON --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM-RTTI,LLVM-COMMON --input-file=%t-cir.ll  %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG-RTTI,OGCG-COMMON --input-file=%t.ll  %s

// Note: This test will be expanded to verify VTT emission and VTT implicit
// argument handling. For now, it's just test the record layout.

class A {
public:
  int a;
  virtual void v();
};

class B : public virtual A {
public:
  int b;
  virtual void w();
};

class C : public virtual A {
public:
  long c;
  virtual void x();
};

class D : public B, public C {
public:
  long d;
  D();
  virtual void y();
};

// This is just here to force the record types to be emitted.
void f(D *d) {}

// Trigger vtable and VTT emission for D.
void D::y() {}

// CIR-COMMON: !rec_A2Ebase = !cir.record<struct "A.base" packed {!cir.vptr, !s32i}>
// CIR-COMMON: !rec_B2Ebase = !cir.record<struct "B.base" packed {!cir.vptr, !s32i}>
// CIR-COMMON: !rec_C2Ebase = !cir.record<struct "C.base" {!cir.vptr, !s64i}>
// CIR-COMMON: !rec_A = !cir.record<class "A" packed padded {!cir.vptr, !s32i, !cir.array<!u8i x 4>}>
// CIR-COMMON: !rec_B = !cir.record<class "B" packed padded {!cir.vptr, !s32i, !cir.array<!u8i x 4>, !rec_A2Ebase, !cir.array<!u8i x 4>}>
// CIR-COMMON: !rec_C = !cir.record<class "C" {!cir.vptr, !s64i, !rec_A2Ebase}>
// CIR-COMMON: !rec_D = !cir.record<class "D" {!rec_B2Ebase, !rec_C2Ebase, !s64i, !rec_A2Ebase}>

// CIR-RTTI:   ![[REC_TYPE_INFO_VTABLE:.*]]= !cir.record<struct  {!cir.ptr<!u8i>, !cir.ptr<!u8i>, !u32i, !u32i, !cir.ptr<!u8i>, !s64i, !cir.ptr<!u8i>, !s64i}>
// CIR-COMMON: ![[REC_D_VTABLE:.*]] = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 5>, !cir.array<!cir.ptr<!u8i> x 4>, !cir.array<!cir.ptr<!u8i> x 4>}>
// CIR-COMMON: ![[REC_B_OR_C_IN_D_VTABLE:.*]]= !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>, !cir.array<!cir.ptr<!u8i> x 4>}>

// Vtable for D

// CIR-COMMON:       cir.global{{.*}} @_ZTV1D = #cir.vtable<{
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<40 : i64> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1D1yEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 5>,
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<24 : i64> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<-16 : i64> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<-40 : i64> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-COMMON-SAME: }> : ![[REC_D_VTABLE]] {alignment = 8 : i64}

// LLVM-COMMON:       @_ZTV1D = global { [5 x ptr], [4 x ptr], [4 x ptr] } {
// LLVM-NO-RTTI-SAME:   [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// LLVM-RTTI-SAME:      [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr @_ZTI1D, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// LLVM-NO-RTTI-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZN1C1xEv],
// LLVM-RTTI-SAME:      [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D, ptr @_ZN1C1xEv],
// LLVM-NO-RTTI-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-RTTI-SAME:      [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr @_ZTI1D, ptr @_ZN1A1vEv]
// LLVM-COMMON-SAME:  }, align 8

// OGCG-COMMON:       @_ZTV1D = unnamed_addr constant { [5 x ptr], [4 x ptr], [4 x ptr] } {
// OGCG-NO-RTTI-SAME:   [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// OGCG-RTTI-SAME:      [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr @_ZTI1D, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZN1C1xEv],
// OGCG-RTTI-SAME:      [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D, ptr @_ZN1C1xEv],
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-RTTI-SAME:      [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr @_ZTI1D, ptr @_ZN1A1vEv]
// OGCG-COMMON-SAME:  }, align 8

// VTT for D

// CIR-COMMON:      cir.global{{.*}} @_ZTT1D = #cir.const_array<[
// CIR-COMMON-SAME:   #cir.global_view<@_ZTV1D, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTC1D0_1B, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTC1D0_1B, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTC1D16_1C, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTC1D16_1C, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTV1D, [2 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:   #cir.global_view<@_ZTV1D, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>
// CIR-COMMON-SAME: ]> : !cir.array<!cir.ptr<!u8i> x 7> {alignment = 8 : i64}

// LLVM-COMMON:      @_ZTT1D = global [7 x ptr] [
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 24),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D0_1B, i64 24),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D0_1B, i64 56),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D16_1C, i64 24),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D16_1C, i64 56),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 96),
// LLVM-COMMON-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 64)
// LLVM-COMMON-SAME: ], align 8

// OGCG-COMMON:      @_ZTT1D = unnamed_addr constant [7 x ptr] [
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D0_1B, i32 0, i32 0, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D0_1B, i32 0, i32 1, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D16_1C, i32 0, i32 0, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D16_1C, i32 0, i32 1, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 2, i32 3),
// OGCG-COMMON-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3)
// OGCG-COMMON-SAME: ], align 8

// Construction vtable for B-in-D

// CIR-COMMON:      cir.global{{.*}} @_ZTC1D0_1B = #cir.vtable<{
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<40 : i64> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<-40 : i64> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-COMMON-SAME:    }> : ![[REC_B_OR_C_IN_D_VTABLE]]

// LLVM-COMMON:       @_ZTC1D0_1B = global { [4 x ptr], [4 x ptr] } {
// LLVM-NO-RTTI-SAME: [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv],
// LLVM-RTTI-SAME:    [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr @_ZTI1B, ptr @_ZN1B1wEv],
// LLVM-NO-RTTI-SAME: [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-RTTI-SAME:    [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr @_ZTI1B, ptr @_ZN1A1vEv]
// LLVM-COMMON-SAME:  }, align 8

// OGCG-COMMON:       @_ZTC1D0_1B = unnamed_addr constant { [4 x ptr], [4 x ptr] } {
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv],
// OGCG-RTTI-SAME:      [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr @_ZTI1B, ptr @_ZN1B1wEv],
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-RTTI-SAME:      [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr @_ZTI1B, ptr @_ZN1A1vEv]
// OGCG-COMMON-SAME:  }, align 8

// CIR-RTTI:  cir.global{{.*}} @_ZTI1B : !cir.ptr<!u8i>

// LLVM-RTTI: @_ZTI1B = external global ptr

// OGCG-RTTI: @_ZTI1B = external constant ptr

// Construction vtable for C-in-D

// CIR-COMMON:       cir.global{{.*}} @_ZTC1D16_1C = #cir.vtable<{
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<24 : i64> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:    ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-COMMON-SAME:    #cir.const_array<[
// CIR-COMMON-SAME:      #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.ptr<-24 : i64> : !cir.ptr<!u8i>,
// CIR-NO-RTTI-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:        #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>,
// CIR-COMMON-SAME:      #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-COMMON-SAME:  ]> : !cir.array<!cir.ptr<!u8i> x 4>}>
// CIR-COMMON-SAME:  : ![[REC_B_OR_C_IN_D_VTABLE]]

// LLVM-COMMON:       @_ZTC1D16_1C = global { [4 x ptr], [4 x ptr] } {
// LLVM-NO-RTTI-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr null, ptr @_ZN1C1xEv],
// LLVM-RTTI-SAME:      [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1C, ptr @_ZN1C1xEv],
// LLVM-NO-RTTI-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-RTTI-SAME:      [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1C, ptr @_ZN1A1vEv]
// LLVM-COMMON-SAME:  }, align 8

// OGCG-COMMON:        @_ZTC1D16_1C = unnamed_addr constant { [4 x ptr], [4 x ptr] } {
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr null, ptr @_ZN1C1xEv],
// OGCG-RTTI-SAME:      [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr @_ZTI1C, ptr @_ZN1C1xEv],
// OGCG-NO-RTTI-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-RTTI-SAME:      [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr @_ZTI1C, ptr @_ZN1A1vEv]
// OGCG-COMMON-SAME:  }, align 8

// RTTI class type info for D

// CIR-RTTI:  cir.globa{{.*}} @_ZTVN10__cxxabiv121__vmi_class_type_infoE : !cir.ptr<!cir.ptr<!u8i>>

// CIR-RTTI:  cir.global{{.*}} @_ZTS1D = #cir.const_array<"1D" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>

// CIR-RTTI:      cir.global{{.*}} @_ZTI1D = #cir.typeinfo<{
// CIR-RTTI-SAME:   #cir.global_view<@_ZTVN10__cxxabiv121__vmi_class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:   #cir.global_view<@_ZTS1D> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:   #cir.int<2> : !u32i, #cir.int<2> : !u32i,
// CIR-RTTI-SAME:   #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:   #cir.int<2> : !s64i,
// CIR-RTTI-SAME:   #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>,
// CIR-RTTI-SAME:   #cir.int<4098> : !s64i}> : !rec_anon_struct

// CIR-RTTI: cir.global{{.*}} @_ZTV1A : !rec_anon_struct3

// LLVM-RTTI: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global ptr
// LLVM-RTTI: @_ZTS1D = global [2 x i8] c"1D", align 1

// LLVM-RTTI:      @_ZTI1D = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } {
// LLVM-RTTI-SAME:   ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 16),
// LLVM-RTTI-SAME:   ptr @_ZTS1D, i32 2, i32 2, ptr @_ZTI1B, i64 2, ptr @_ZTI1C, i64 4098 }

// OGCG-RTTI:      @_ZTI1D = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } {
// OGCG-RTTI-SAME:   ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2),
// OGCG-RTTI-SAME:   ptr @_ZTS1D, i32 2, i32 2, ptr @_ZTI1B, i64 2, ptr @_ZTI1C, i64 4098 }, align 8

// OGCG-RTTI: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// OGCG-RTTI: @_ZTS1D = constant [3 x i8] c"1D\00", align 1
// OGCG-RTTI: @_ZTV1A = external unnamed_addr constant { [3 x ptr] }, align 8

D::D() {}

// In CIR, this gets emitted after the B and C constructors. See below.
// Base (C2) constructor for D

// OGCG-COMMON: define {{.*}} void @_ZN1DC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[B_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG-COMMON:   call void @_ZN1BC2Ev(ptr {{.*}} %[[THIS]], ptr {{.*}} %[[B_VTT]])
// OGCG-COMMON:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG-COMMON:   %[[C_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 3
// OGCG-COMMON:   call void @_ZN1CC2Ev(ptr {{.*}} %[[C_ADDR]], ptr {{.*}} %[[C_VTT]])
// OGCG-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG-COMMON:   %[[D_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 5
// OGCG-COMMON:   %[[D_VPTR:.*]] = load ptr, ptr %[[D_VPTR_ADDR]]
// OGCG-COMMON:   %[[D_VPTR_ADDR2:.*]] = load ptr, ptr %[[THIS]]
// OGCG-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[D_VPTR_ADDR2]], i64 -24
// OGCG-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG-COMMON:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG-COMMON:   store ptr %[[D_VPTR]], ptr %[[BASE_PTR]]
// OGCG-COMMON:   %[[C_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 6
// OGCG-COMMON:   %[[C_VPTR:.*]] = load ptr, ptr %[[C_VPTR_ADDR]]
// OGCG-COMMON:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG-COMMON:   store ptr %[[C_VPTR]], ptr %[[C_ADDR]]

// Base (C2) constructor for B

// CIR-COMMON:      cir.func {{.*}} @_ZN1BC2Ev
// CIR-COMMON-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_B>
// CIR-COMMON-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR-COMMON:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR-COMMON:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR-COMMON:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR-COMMON:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-COMMON:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR-COMMON:        %[[VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[VPTR_ADDR:.*]] = cir.cast bitcast %[[VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]]
// CIR-COMMON:        %[[B_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR-COMMON:        cir.store{{.*}} %[[VPTR]], %[[B_VPTR_ADDR]]
// CIR-COMMON:        %[[B_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[B_VPTR_ADDR:.*]] = cir.cast bitcast %[[B_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[B_VPTR:.*]] = cir.load{{.*}} %[[B_VPTR_ADDR]]
// CIR-COMMON:        %[[B_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR-COMMON:        %[[VPTR:.*]] = cir.load{{.*}} %[[B_VPTR_ADDR]]
// CIR-COMMON:        %[[VPTR_ADDR2:.*]] = cir.cast bitcast %[[VPTR]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[CONST_24:.*]] = cir.const #cir.int<-24>
// CIR-COMMON:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride %[[VPTR_ADDR2]], %[[CONST_24]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_OFFSET_PTR:.*]] = cir.cast bitcast %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR-COMMON:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR-COMMON:        %[[THIS_PTR:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_PTR:.*]] = cir.ptr_stride %[[THIS_PTR]], %[[BASE_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_CAST:.*]] = cir.cast bitcast %[[BASE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_B>
// CIR-COMMON:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR-COMMON:        cir.store{{.*}} %[[B_VPTR]], %[[BASE_VPTR_ADDR]]

// LLVM-COMMON: define {{.*}} void @_ZN1BC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]])
// LLVM-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM-COMMON:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM-COMMON:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// LLVM-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// LLVM-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM-COMMON:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM-COMMON:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// OGCG-COMMON: define {{.*}} void @_ZN1BC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG-COMMON:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG-COMMON:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// OGCG-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// OGCG-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG-COMMON:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG-COMMON:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// Base (C2) constructor for C

// CIR-COMMON:      cir.func {{.*}} @_ZN1CC2Ev
// CIR-COMMON-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_C>
// CIR-COMMON-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR-COMMON:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR-COMMON:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR-COMMON:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR-COMMON:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-COMMON:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR-COMMON:        %[[VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[VPTR_ADDR:.*]] = cir.cast bitcast %[[VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]]
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR-COMMON:        cir.store{{.*}} %[[VPTR]], %[[C_VPTR_ADDR]]
// CIR-COMMON:        %[[C_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.cast bitcast %[[C_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[C_VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]]
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR-COMMON:        %[[VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]]
// CIR-COMMON:        %[[VPTR_ADDR2:.*]] = cir.cast bitcast %[[VPTR]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[CONST_24:.*]] = cir.const #cir.int<-24>
// CIR-COMMON:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride %[[VPTR_ADDR2]], %[[CONST_24]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_OFFSET_PTR:.*]] = cir.cast bitcast %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR-COMMON:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR-COMMON:        %[[THIS_PTR:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_C> -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_PTR:.*]] = cir.ptr_stride %[[THIS_PTR]], %[[BASE_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_CAST:.*]] = cir.cast bitcast %[[BASE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_C>
// CIR-COMMON:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR-COMMON:        cir.store{{.*}} %[[C_VPTR]], %[[BASE_VPTR_ADDR]]

// LLVM-COMMON: define {{.*}} void @_ZN1CC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]])
// LLVM-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM-COMMON:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM-COMMON:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// LLVM-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// LLVM-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM-COMMON:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM-COMMON:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// OGCG-COMMON: define {{.*}} void @_ZN1CC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG-COMMON:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG-COMMON:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// OGCG-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// OGCG-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG-COMMON:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG-COMMON:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// Base (C2) constructor for D

// CIR-COMMON:      cir.func {{.*}} @_ZN1DC2Ev
// CIR-COMMON-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_D>
// CIR-COMMON-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR-COMMON:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR-COMMON:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR-COMMON:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR-COMMON:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-COMMON:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR-COMMON:        %[[B_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_B>
// CIR-COMMON:        %[[B_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        cir.call @_ZN1BC2Ev(%[[B_ADDR]], %[[B_VTT]]) nothrow : (!cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR-COMMON:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR-COMMON:        %[[C_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        cir.call @_ZN1CC2Ev(%[[C_ADDR]], %[[C_VTT]]) nothrow : (!cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR-COMMON:        %[[D_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[VPTR_ADDR:.*]] = cir.cast bitcast %[[D_VTT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-COMMON:        %[[D_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR-COMMON:        cir.store{{.*}} %[[VPTR]], %[[D_VPTR_ADDR]]
// CIR-COMMON:        %[[D_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 5 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[D_VPTR_ADDR:.*]] = cir.cast bitcast %[[D_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[D_VPTR:.*]] = cir.load{{.*}} %[[D_VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-COMMON:        %[[D_VPTR_ADDR2:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[VPTR2:.*]] = cir.load{{.*}} %[[D_VPTR_ADDR2]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-COMMON:        %[[VPTR_ADDR2:.*]] = cir.cast bitcast %[[VPTR2]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[CONST_24:.*]] = cir.const #cir.int<-24> : !s64i
// CIR-COMMON:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride %[[VPTR_ADDR2]], %[[CONST_24]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_OFFSET_PTR:.*]] = cir.cast bitcast %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR-COMMON:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR-COMMON:        %[[THIS_PTR:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_D> -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_PTR:.*]] = cir.ptr_stride %[[THIS_PTR]], %[[BASE_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-COMMON:        %[[BASE_CAST:.*]] = cir.cast bitcast %[[BASE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_D>
// CIR-COMMON:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR-COMMON:        cir.store{{.*}} %[[D_VPTR]], %[[BASE_VPTR_ADDR]]
// CIR-COMMON:        %[[C_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 6 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.cast bitcast %[[C_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[C_VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-COMMON:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[C_ADDR]] : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        cir.store{{.*}} %[[C_VPTR]], %[[C_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM-COMMON: define {{.*}} void @_ZN1DC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]]){{.*}} {
// LLVM-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM-COMMON:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-COMMON:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM-COMMON:   %[[B_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM-COMMON:   call void @_ZN1BC2Ev(ptr %[[THIS]], ptr %[[B_VTT]])
// LLVM-COMMON:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM-COMMON:   %[[C_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 3
// LLVM-COMMON:   call void @_ZN1CC2Ev(ptr %[[C_ADDR]], ptr %[[C_VTT]])
// LLVM-COMMON:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM-COMMON:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM-COMMON:   %[[D_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 5
// LLVM-COMMON:   %[[D_VPTR:.*]] = load ptr, ptr %[[D_VPTR_ADDR]]
// LLVM-COMMON:   %[[D_VPTR_ADDR2:.*]] = load ptr, ptr %[[THIS]]
// LLVM-COMMON:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[D_VPTR_ADDR2]], i64 -24
// LLVM-COMMON:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM-COMMON:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM-COMMON:   store ptr %[[D_VPTR]], ptr %[[BASE_PTR]]
// LLVM-COMMON:   %[[C_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 6
// LLVM-COMMON:   %[[C_VPTR:.*]] = load ptr, ptr %[[C_VPTR_ADDR]]
// LLVM-COMMON:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM-COMMON:   store ptr %[[C_VPTR]], ptr %[[C_ADDR]]

// The C2 constructor for D gets emitted earlier in OGCG, see above.

// Base (C2) constructor for A

// CIR-COMMON:      cir.func {{.*}} @_ZN1AC2Ev
// CIR-COMMON-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_A>
// CIR-COMMON:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR-COMMON:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR-COMMON:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-COMMON:        %[[VPTR:.*]] = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-COMMON:        %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        cir.store{{.*}} %[[VPTR]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM-COMMON: define {{.*}} void @_ZN1AC2Ev(ptr %[[THIS_ARG:.*]]){{.*}} {
// LLVM-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]], align 8
// LLVM-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// LLVM-COMMON:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 16), ptr %[[THIS]]

// The C2 constructor for A gets emitted later in OGCG, see below.

// Complete (C1) constructor for D

// CIR-COMMON:      cir.func {{.*}} @_ZN1DC1Ev
// CIR-COMMON-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_D>
// CIR-COMMON:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR-COMMON:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR-COMMON:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-COMMON:        %[[A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR-COMMON:        cir.call @_ZN1AC2Ev(%[[A_ADDR]]) nothrow : (!cir.ptr<!rec_A>) -> ()
// CIR-COMMON:        %[[B_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_B>
// CIR-COMMON:        %[[B_VTT:.*]] = cir.vtt.address_point @_ZTT1D, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        cir.call @_ZN1BC2Ev(%[[B_ADDR]], %[[B_VTT]]) nothrow : (!cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR-COMMON:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR-COMMON:        %[[C_VTT:.*]] = cir.vtt.address_point @_ZTT1D, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR-COMMON:        cir.call @_ZN1CC2Ev(%[[C_ADDR]], %[[C_VTT]]) nothrow : (!cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR-COMMON:        %[[D_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR-COMMON:        %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        cir.store{{.*}} %[[D_VPTR]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[A_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 2, offset = 3>) : !cir.vptr
// CIR-COMMON:        %[[A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR-COMMON:        %[[A_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[A_ADDR]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        cir.store{{.*}} %[[A_VPTR]], %[[A_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-COMMON:        %[[C_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 1, offset = 3>) : !cir.vptr
// CIR-COMMON:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR-COMMON:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[C_ADDR]] : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR-COMMON:        cir.store{{.*}} %[[C_VPTR]], %[[C_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM-COMMON: define {{.*}} void @_ZN1DC1Ev(ptr %[[THIS_ARG:.*]])
// LLVM-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-COMMON:   %[[A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 40
// LLVM-COMMON:   call void @_ZN1AC2Ev(ptr %[[A_ADDR]])
// LLVM-COMMON:   call void @_ZN1BC2Ev(ptr %[[THIS]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 8))
// LLVM-COMMON:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM-COMMON:   call void @_ZN1CC2Ev(ptr %[[C_ADDR]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 24))
// LLVM-COMMON:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 24), ptr %[[THIS]]
// LLVM-COMMON:   %[[A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 40
// LLVM-COMMON:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 96), ptr %[[A_ADDR]]
// LLVM-COMMON:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM-COMMON:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 64), ptr %[[C_ADDR]]

// OGCG-COMMON: define {{.*}} void @_ZN1DC1Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 40
// OGCG-COMMON:   call void @_ZN1AC2Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG-COMMON:   call void @_ZN1BC2Ev(ptr {{.*}} %[[THIS]], ptr {{.*}} getelementptr inbounds ([7 x ptr], ptr @_ZTT1D, i64 0, i64 1))
// OGCG-COMMON:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG-COMMON:   call void @_ZN1CC2Ev(ptr {{.*}} %[[C_ADDR]], ptr {{.*}} getelementptr inbounds ([7 x ptr], ptr @_ZTT1D, i64 0, i64 3))
// OGCG-COMMON:   store ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), ptr %[[THIS]]
// OGCG-COMMON:   %[[A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 40
// OGCG-COMMON:   store ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 2, i32 3), ptr %[[A_ADDR]]
// OGCG-COMMON:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG-COMMON:   store ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3), ptr %[[C_ADDR]]

// OGCG-COMMON: define {{.*}} void @_ZN1AC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG-COMMON:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG-COMMON:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG-COMMON:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG-COMMON:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %[[THIS]]
