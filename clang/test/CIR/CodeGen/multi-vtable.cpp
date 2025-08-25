// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fno-rtti -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fno-rtti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Note: This test is using -fno-rtti so that we can delay implemntation of that handling.
//       When rtti handling for vtables is implemented, that option should be removed.

class Mother {
public:
  virtual void MotherKey();
  void simple() { }
  virtual void MotherNonKey() {}
};

class Father {
public:
  virtual void FatherKey();
};

class Child : public Mother, public Father {
public:
  void MotherKey() override;
};

void Mother::MotherKey() {}
void Father::FatherKey() {}
void Child::MotherKey() {}

// CIR-DAG: [[MOTHER_VTABLE_TYPE:.*]] = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>}>
// CIR-DAG: [[FATHER_VTABLE_TYPE:.*]] = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 3>}>
// CIR-DAG: [[CHILD_VTABLE_TYPE:.*]] = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 4>, !cir.array<!cir.ptr<!u8i> x 3>}>
// CIR-DAG: !rec_Father = !cir.record<class "Father" {!cir.vptr}
// CIR-DAG: !rec_Mother = !cir.record<class "Mother" {!cir.vptr}
// CIR-DAG: !rec_Child = !cir.record<class "Child" {!rec_Mother, !rec_Father}

// Mother vtable

// CIR:      cir.global "private" external @_ZTV6Mother = #cir.vtable<{
// CIR-SAME:      #cir.const_array<[
// CIR-SAME:          #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:          #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:          #cir.global_view<@_ZN6Mother9MotherKeyEv> : !cir.ptr<!u8i>,
// CIR-SAME:          #cir.global_view<@_ZN6Mother12MotherNonKeyEv> : !cir.ptr<!u8i>
// CIR-SAME:      ]> : !cir.array<!cir.ptr<!u8i> x 4>
// CIR-SAME: }> : [[MOTHER_VTABLE_TYPE]]

// LLVM:      @_ZTV6Mother = global { [4 x ptr] } {
// LLVM-SAME:     [4 x ptr] [
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr @_ZN6Mother9MotherKeyEv,
// LLVM-SAME:         ptr @_ZN6Mother12MotherNonKeyEv
// LLVM-SAME:     ]
// LLVM-SAME: }

// OGCG:      @_ZTV6Mother = unnamed_addr constant { [4 x ptr] } {
// OGCG-SAME:     [4 x ptr] [
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr @_ZN6Mother9MotherKeyEv,
// OGCG-SAME:         ptr @_ZN6Mother12MotherNonKeyEv
// OGCG-SAME:     ]
// OGCG-SAME: }

// Father vtable

// CIR:      cir.global "private" external @_ZTV6Father = #cir.vtable<{
// CIR-SAME:     #cir.const_array<[
// CIR-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.global_view<@_ZN6Father9FatherKeyEv> : !cir.ptr<!u8i>
// CIR-SAME:     ]> : !cir.array<!cir.ptr<!u8i> x 3>
// CIR-SAME: }> : [[FATHER_VTABLE_TYPE]]

// LLVM:      @_ZTV6Father = global { [3 x ptr] } {
// LLVM-SAME:     [3 x ptr] [
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr @_ZN6Father9FatherKeyEv
// LLVM-SAME:     ]
// LLVM-SAME: }

// OGCG:      @_ZTV6Father = unnamed_addr constant { [3 x ptr] } {
// OGCG-SAME:     [3 x ptr] [
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr @_ZN6Father9FatherKeyEv
// OGCG-SAME:     ]
// OGCG-SAME: }

// Child vtable

// CIR:      cir.global "private" external @_ZTV5Child = #cir.vtable<{
// CIR-SAME:     #cir.const_array<[
// CIR-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.global_view<@_ZN5Child9MotherKeyEv> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.global_view<@_ZN6Mother12MotherNonKeyEv> : !cir.ptr<!u8i>
// CIR-SAME:     ]> : !cir.array<!cir.ptr<!u8i> x 4>,
// CIR-SAME:     #cir.const_array<[
// CIR-SAME:         #cir.ptr<-8 : i64> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:         #cir.global_view<@_ZN6Father9FatherKeyEv> : !cir.ptr<!u8i>
// CIR-SAME:     ]> : !cir.array<!cir.ptr<!u8i> x 3>
// CIR-SAME: }> : [[CHILD_VTABLE_TYPE]]

// LLVM:      @_ZTV5Child = global { [4 x ptr], [3 x ptr] } {
// LLVM-SAME:     [4 x ptr] [
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr @_ZN5Child9MotherKeyEv,
// LLVM-SAME:         ptr @_ZN6Mother12MotherNonKeyEv
// LLVM-SAME:     ],
// LLVM-SAME:     [3 x ptr] [
// LLVM-SAME:         ptr inttoptr (i64 -8 to ptr),
// LLVM-SAME:         ptr null,
// LLVM-SAME:         ptr @_ZN6Father9FatherKeyEv
// LLVM-SAME:     ]
// LLVM-SAME: }

// OGCG:      @_ZTV5Child = unnamed_addr constant { [4 x ptr], [3 x ptr] } {
// OGCG-SAME:     [4 x ptr] [
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr @_ZN5Child9MotherKeyEv,
// OGCG-SAME:         ptr @_ZN6Mother12MotherNonKeyEv
// OGCG-SAME:     ],
// OGCG-SAME:     [3 x ptr] [
// OGCG-SAME:         ptr inttoptr (i64 -8 to ptr),
// OGCG-SAME:         ptr null,
// OGCG-SAME:         ptr @_ZN6Father9FatherKeyEv
// OGCG-SAME:     ]
// OGCG-SAME: }
