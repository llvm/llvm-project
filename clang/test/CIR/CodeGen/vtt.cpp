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
