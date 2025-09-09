// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll  %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll  %s

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

// CIR: !rec_A2Ebase = !cir.record<struct "A.base" packed {!cir.vptr, !s32i}>
// CIR: !rec_B2Ebase = !cir.record<struct "B.base" packed {!cir.vptr, !s32i}>
// CIR: !rec_C2Ebase = !cir.record<struct "C.base" {!cir.vptr, !s64i}>
// CIR: !rec_A = !cir.record<class "A" packed padded {!cir.vptr, !s32i, !cir.array<!u8i x 4>}>
// CIR: !rec_B = !cir.record<class "B" packed padded {!cir.vptr, !s32i, !cir.array<!u8i x 4>, !rec_A2Ebase, !cir.array<!u8i x 4>}>
// CIR: !rec_C = !cir.record<class "C" {!cir.vptr, !s64i, !rec_A2Ebase}>
// CIR: !rec_D = !cir.record<class "D" {!rec_B2Ebase, !rec_C2Ebase, !s64i, !rec_A2Ebase}>

// CIR: !rec_anon_struct = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 5>, !cir.array<!cir.ptr<!u8i> x 4>, !cir.array<!cir.ptr<!u8i> x 4>}>
// CIR: !rec_anon_struct1 = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>, !cir.array<!cir.ptr<!u8i> x 4>}>

// Vtable for D
// CIR:      cir.global{{.*}} @_ZTV1D = #cir.vtable<{
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<40 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1D1yEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 5>,
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<24 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<-16 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<-40 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-SAME: }> : !rec_anon_struct {alignment = 8 : i64}

// LLVM:      @_ZTV1D = global { [5 x ptr], [4 x ptr], [4 x ptr] } {
// LLVM-SAME:   [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// LLVM-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZN1C1xEv],
// LLVM-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-SAME: }, align 8

// OGCG:      @_ZTV1D = unnamed_addr constant { [5 x ptr], [4 x ptr], [4 x ptr] } {
// OGCG-SAME:   [5 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv, ptr @_ZN1D1yEv],
// OGCG-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZN1C1xEv],
// OGCG-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-SAME: }, align 8

// VTT for D
// CIR:      cir.global{{.*}} @_ZTT1D = #cir.const_array<[
// CIR-SAME:   #cir.global_view<@_ZTV1D, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTC1D0_1B, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTC1D0_1B, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTC1D16_1C, [0 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTC1D16_1C, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTV1D, [2 : i32, 3 : i32]> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZTV1D, [1 : i32, 3 : i32]> : !cir.ptr<!u8i>
// CIR-SAME: ]> : !cir.array<!cir.ptr<!u8i> x 7> {alignment = 8 : i64}

// LLVM:      @_ZTT1D = global [7 x ptr] [
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 24),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D0_1B, i64 24),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D0_1B, i64 56),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D16_1C, i64 24),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTC1D16_1C, i64 56),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 96),
// LLVM-SAME:   ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 64)
// LLVM-SAME: ], align 8

// OGCG:      @_ZTT1D = unnamed_addr constant [7 x ptr] [
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D0_1B, i32 0, i32 0, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D0_1B, i32 0, i32 1, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D16_1C, i32 0, i32 0, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTC1D16_1C, i32 0, i32 1, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 2, i32 3),
// OGCG-SAME:   ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3)
// OGCG-SAME: ], align 8

// Construction vtable for B-in-D
// CIR:      cir.global{{.*}} @_ZTC1D0_1B = #cir.vtable<{
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<40 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1B1wEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<-40 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-SAME: }> : !rec_anon_struct1 {alignment = 8 : i64}

// LLVM:      @_ZTC1D0_1B = global { [4 x ptr], [4 x ptr] } {
// LLVM-SAME:   [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv],
// LLVM-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-SAME: }, align 8

// OGCG:      @_ZTC1D0_1B = unnamed_addr constant { [4 x ptr], [4 x ptr] } {
// OGCG-SAME:   [4 x ptr] [ptr inttoptr (i64 40 to ptr), ptr null, ptr null, ptr @_ZN1B1wEv],
// OGCG-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -40 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-SAME: }, align 8

// Construction vtable for C-in-D
// CIR:      cir.global{{.*}} @_ZTC1D16_1C = #cir.vtable<{
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<24 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1C1xEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-SAME:   #cir.const_array<[
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<-24 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:     #cir.global_view<@_ZN1A1vEv> : !cir.ptr<!u8i>
// CIR-SAME:   ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-SAME: }> : !rec_anon_struct1 {alignment = 8 : i64}

// LLVM:      @_ZTC1D16_1C = global { [4 x ptr], [4 x ptr] } {
// LLVM-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr null, ptr @_ZN1C1xEv],
// LLVM-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr null, ptr @_ZN1A1vEv]
// LLVM-SAME: }, align 8

// OGCG:      @_ZTC1D16_1C = unnamed_addr constant { [4 x ptr], [4 x ptr] } {
// OGCG-SAME:   [4 x ptr] [ptr inttoptr (i64 24 to ptr), ptr null, ptr null, ptr @_ZN1C1xEv],
// OGCG-SAME:   [4 x ptr] [ptr null, ptr inttoptr (i64 -24 to ptr), ptr null, ptr @_ZN1A1vEv]
// OGCG-SAME: }, align 8

D::D() {}

// In CIR, this gets emitted after the B and C constructors. See below.
// Base (C2) constructor for D

// OGCG: define {{.*}} void @_ZN1DC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   %[[B_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG:   call void @_ZN1BC2Ev(ptr {{.*}} %[[THIS]], ptr {{.*}} %[[B_VTT]])
// OGCG:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG:   %[[C_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 3
// OGCG:   call void @_ZN1CC2Ev(ptr {{.*}} %[[C_ADDR]], ptr {{.*}} %[[C_VTT]])
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG:   %[[D_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 5
// OGCG:   %[[D_VPTR:.*]] = load ptr, ptr %[[D_VPTR_ADDR]]
// OGCG:   %[[D_VPTR_ADDR2:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[D_VPTR_ADDR2]], i64 -24
// OGCG:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG:   store ptr %[[D_VPTR]], ptr %[[BASE_PTR]]
// OGCG:   %[[C_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 6
// OGCG:   %[[C_VPTR:.*]] = load ptr, ptr %[[C_VPTR_ADDR]]
// OGCG:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG:   store ptr %[[C_VPTR]], ptr %[[C_ADDR]]


// Base (C2) constructor for B

// CIR:      cir.func {{.*}} @_ZN1BC2Ev
// CIR-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_B>
// CIR-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR:        %[[VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[VPTR_ADDR:.*]] = cir.cast(bitcast, %[[VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]]
// CIR:        %[[B_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR:        cir.store{{.*}} %[[VPTR]], %[[B_VPTR_ADDR]]
// CIR:        %[[B_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[B_VPTR_ADDR:.*]] = cir.cast(bitcast, %[[B_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[B_VPTR:.*]] = cir.load{{.*}} %[[B_VPTR_ADDR]]
// CIR:        %[[B_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[B_VPTR_ADDR]]
// CIR:        %[[VPTR_ADDR2:.*]] = cir.cast(bitcast, %[[VPTR]] : !cir.vptr), !cir.ptr<!u8i>
// CIR:        %[[CONST_24:.*]] = cir.const #cir.int<-24>
// CIR:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride(%[[VPTR_ADDR2]] : !cir.ptr<!u8i>, %[[CONST_24]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_OFFSET_PTR:.*]] = cir.cast(bitcast, %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR:        %[[THIS_PTR:.*]] = cir.cast(bitcast, %[[THIS]] : !cir.ptr<!rec_B>), !cir.ptr<!u8i>
// CIR:        %[[BASE_PTR:.*]] = cir.ptr_stride(%[[THIS_PTR]] : !cir.ptr<!u8i>, %[[BASE_OFFSET]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_CAST:.*]] = cir.cast(bitcast, %[[BASE_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!rec_B>
// CIR:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR:        cir.store{{.*}} %[[B_VPTR]], %[[BASE_VPTR_ADDR]]

// LLVM: define {{.*}} void @_ZN1BC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// LLVM:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// OGCG: define {{.*}} void @_ZN1BC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// OGCG:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// Base (C2) constructor for C

// CIR:      cir.func {{.*}} @_ZN1CC2Ev
// CIR-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_C>
// CIR-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR:        %[[VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[VPTR_ADDR:.*]] = cir.cast(bitcast, %[[VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]]
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR:        cir.store{{.*}} %[[VPTR]], %[[C_VPTR_ADDR]]
// CIR:        %[[C_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.cast(bitcast, %[[C_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[C_VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]]
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]]
// CIR:        %[[VPTR_ADDR2:.*]] = cir.cast(bitcast, %[[VPTR]] : !cir.vptr), !cir.ptr<!u8i>
// CIR:        %[[CONST_24:.*]] = cir.const #cir.int<-24>
// CIR:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride(%[[VPTR_ADDR2]] : !cir.ptr<!u8i>, %[[CONST_24]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_OFFSET_PTR:.*]] = cir.cast(bitcast, %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR:        %[[THIS_PTR:.*]] = cir.cast(bitcast, %[[THIS]] : !cir.ptr<!rec_C>), !cir.ptr<!u8i>
// CIR:        %[[BASE_PTR:.*]] = cir.ptr_stride(%[[THIS_PTR]] : !cir.ptr<!u8i>, %[[BASE_OFFSET]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_CAST:.*]] = cir.cast(bitcast, %[[BASE_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!rec_C>
// CIR:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR:        cir.store{{.*}} %[[C_VPTR]], %[[BASE_VPTR_ADDR]]

// LLVM: define {{.*}} void @_ZN1CC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// LLVM:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// OGCG: define {{.*}} void @_ZN1CC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG:   %[[B_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG:   %[[B_VPTR:.*]] = load ptr, ptr %[[B_VPTR_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -24
// OGCG:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG:   store ptr %[[B_VPTR]], ptr %[[BASE_PTR]]

// Base (C2) constructor for D

// CIR:      cir.func {{.*}} @_ZN1DC2Ev
// CIR-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_D>
// CIR-SAME:                      %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR:        %[[B_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_B>
// CIR:        %[[B_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        cir.call @_ZN1BC2Ev(%[[B_ADDR]], %[[B_VTT]]) nothrow : (!cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:        %[[C_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        cir.call @_ZN1CC2Ev(%[[C_ADDR]], %[[C_VTT]]) nothrow : (!cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR:        %[[D_VTT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[VPTR_ADDR:.*]] = cir.cast(bitcast, %[[D_VTT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[D_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]]
// CIR:        cir.store{{.*}} %[[VPTR]], %[[D_VPTR_ADDR]]
// CIR:        %[[D_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 5 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[D_VPTR_ADDR:.*]] = cir.cast(bitcast, %[[D_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[D_VPTR:.*]] = cir.load{{.*}} %[[D_VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[D_VPTR_ADDR2:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR2:.*]] = cir.load{{.*}} %[[D_VPTR_ADDR2]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[VPTR_ADDR2:.*]] = cir.cast(bitcast, %[[VPTR2]] : !cir.vptr), !cir.ptr<!u8i>
// CIR:        %[[CONST_24:.*]] = cir.const #cir.int<-24> : !s64i
// CIR:        %[[BASE_OFFSET_ADDR:.*]] = cir.ptr_stride(%[[VPTR_ADDR2]] : !cir.ptr<!u8i>, %[[CONST_24]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_OFFSET_PTR:.*]] = cir.cast(bitcast, %[[BASE_OFFSET_ADDR]] : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_PTR]] : !cir.ptr<!s64i>, !s64i
// CIR:        %[[THIS_PTR:.*]] = cir.cast(bitcast, %[[THIS]] : !cir.ptr<!rec_D>), !cir.ptr<!u8i>
// CIR:        %[[BASE_PTR:.*]] = cir.ptr_stride(%[[THIS_PTR]] : !cir.ptr<!u8i>, %[[BASE_OFFSET]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_CAST:.*]] = cir.cast(bitcast, %[[BASE_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!rec_D>
// CIR:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_CAST]]
// CIR:        cir.store{{.*}} %[[D_VPTR]], %[[BASE_VPTR_ADDR]]
// CIR:        %[[C_VTT_ADDR_POINT:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 6 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.cast(bitcast, %[[C_VTT_ADDR_POINT]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[C_VPTR:.*]] = cir.load{{.*}} %[[C_VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[C_ADDR]] : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[C_VPTR]], %[[C_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM: define {{.*}} void @_ZN1DC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]]) {
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   %[[B_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM:   call void @_ZN1BC2Ev(ptr %[[THIS]], ptr %[[B_VTT]])
// LLVM:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM:   %[[C_VTT:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 3
// LLVM:   call void @_ZN1CC2Ev(ptr %[[C_ADDR]], ptr %[[C_VTT]])
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM:   %[[D_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 5
// LLVM:   %[[D_VPTR:.*]] = load ptr, ptr %[[D_VPTR_ADDR]]
// LLVM:   %[[D_VPTR_ADDR2:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[D_VPTR_ADDR2]], i64 -24
// LLVM:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM:   store ptr %[[D_VPTR]], ptr %[[BASE_PTR]]
// LLVM:   %[[C_VPTR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 6
// LLVM:   %[[C_VPTR:.*]] = load ptr, ptr %[[C_VPTR_ADDR]]
// LLVM:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM:   store ptr %[[C_VPTR]], ptr %[[C_ADDR]]

// The C2 constructor for D gets emitted earlier in OGCG, see above.

// Base (C2) constructor for A

// CIR:      cir.func {{.*}} @_ZN1AC2Ev
// CIR-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_A>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[VPTR:.*]] = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:        %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[VPTR]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM: define {{.*}} void @_ZN1AC2Ev(ptr %[[THIS_ARG:.*]]) {
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]], align 8
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 16), ptr %[[THIS]]

// The C2 constructor for A gets emitted later in OGCG, see below.

// Complete (C1) constructor for D

// CIR:      cir.func {{.*}} @_ZN1DC1Ev
// CIR-SAME:                      %[[THIS_ARG:.*]]: !cir.ptr<!rec_D>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR:        cir.call @_ZN1AC2Ev(%[[A_ADDR]]) nothrow : (!cir.ptr<!rec_A>) -> ()
// CIR:        %[[B_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_B>
// CIR:        %[[B_VTT:.*]] = cir.vtt.address_point @_ZTT1D, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        cir.call @_ZN1BC2Ev(%[[B_ADDR]], %[[B_VTT]]) nothrow : (!cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:        %[[C_VTT:.*]] = cir.vtt.address_point @_ZTT1D, offset = 3 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        cir.call @_ZN1CC2Ev(%[[C_ADDR]], %[[C_VTT]]) nothrow : (!cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!void>>) -> ()
// CIR:        %[[D_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 0, offset = 3>) : !cir.vptr
// CIR:        %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[D_VPTR]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:        %[[A_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 2, offset = 3>) : !cir.vptr
// CIR:        %[[A_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [40] -> !cir.ptr<!rec_A>
// CIR:        %[[A_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[A_ADDR]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[A_VPTR]], %[[A_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:        %[[C_VPTR:.*]] = cir.vtable.address_point(@_ZTV1D, address_point = <index = 1, offset = 3>) : !cir.vptr
// CIR:        %[[C_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_D> nonnull [16] -> !cir.ptr<!rec_C>
// CIR:        %[[C_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[C_ADDR]] : !cir.ptr<!rec_C> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[C_VPTR]], %[[C_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>

// LLVM: define {{.*}} void @_ZN1DC1Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 40
// LLVM:   call void @_ZN1AC2Ev(ptr %[[A_ADDR]])
// LLVM:   call void @_ZN1BC2Ev(ptr %[[THIS]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 8))
// LLVM:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM:   call void @_ZN1CC2Ev(ptr %[[C_ADDR]], ptr getelementptr inbounds nuw (i8, ptr @_ZTT1D, i64 24))
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 24), ptr %[[THIS]]
// LLVM:   %[[A_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 40
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 96), ptr %[[A_ADDR]]
// LLVM:   %[[C_ADDR:.*]] = getelementptr i8, ptr %[[THIS]], i32 16
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 64), ptr %[[C_ADDR]]

// OGCG: define {{.*}} void @_ZN1DC1Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 40
// OGCG:   call void @_ZN1AC2Ev(ptr {{.*}} %[[A_ADDR]])
// OGCG:   call void @_ZN1BC2Ev(ptr {{.*}} %[[THIS]], ptr {{.*}} getelementptr inbounds ([7 x ptr], ptr @_ZTT1D, i64 0, i64 1))
// OGCG:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG:   call void @_ZN1CC2Ev(ptr {{.*}} %[[C_ADDR]], ptr {{.*}} getelementptr inbounds ([7 x ptr], ptr @_ZTT1D, i64 0, i64 3))
// OGCG:   store ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), ptr %[[THIS]]
// OGCG:   %[[A_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 40
// OGCG:   store ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 2, i32 3), ptr %[[A_ADDR]]
// OGCG:   %[[C_ADDR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 16
// OGCG:   store ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr], [4 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3), ptr %[[C_ADDR]]

// OGCG: define {{.*}} void @_ZN1AC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %[[THIS]]
