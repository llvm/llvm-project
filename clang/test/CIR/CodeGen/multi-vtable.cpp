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
  Child();
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


Child::Child() {}

// CIR: cir.func {{.*}} @_ZN5ChildC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Child>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[MOTHER_BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Child> nonnull [0] -> !cir.ptr<!rec_Mother>
// CIR:   cir.call @_ZN6MotherC2Ev(%[[MOTHER_BASE]]) nothrow : (!cir.ptr<!rec_Mother>) -> ()
// CIR:   %[[FATHER_BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Child> nonnull [8] -> !cir.ptr<!rec_Father>
// CIR:   cir.call @_ZN6FatherC2Ev(%[[FATHER_BASE]]) nothrow : (!cir.ptr<!rec_Father>) -> ()
// CIR:   %[[CHILD_VPTR:.*]] = cir.vtable.address_point(@_ZTV5Child, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:   %[[CHILD_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Child> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %[[CHILD_VPTR]], %[[CHILD_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %[[FATHER_IN_CHILD_VPTR:.*]] = cir.vtable.address_point(@_ZTV5Child, address_point = <index = 1, offset = 2>) : !cir.vptr
// CIR:   %[[FATHER_BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Child> nonnull [8] -> !cir.ptr<!rec_Father>
// CIR:   %[[FATHER_IN_CHILD_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[FATHER_BASE]] : !cir.ptr<!rec_Father> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %[[FATHER_IN_CHILD_VPTR]], %[[FATHER_IN_CHILD_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   cir.return

// The GEP instructions are different between LLVM and OGCG, but they calculate the same addresses.

// LLVM: define{{.*}} void @_ZN5ChildC2Ev(ptr{{.*}} %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   call void @_ZN6MotherC2Ev(ptr{{.*}} %[[THIS]])
// LLVM:   %[[FATHER_BASE:.*]] = getelementptr{{.*}} i8, ptr %[[THIS]], i32 8
// LLVM:   call void @_ZN6FatherC2Ev(ptr{{.*}} %[[FATHER_BASE]])
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV5Child, i64 16), ptr %[[THIS]]
// LLVM:   %[[FATHER_BASE:.*]] = getelementptr{{.*}} i8, ptr %[[THIS]], i32 8
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV5Child, i64 48), ptr %[[FATHER_BASE]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_ZN5ChildC2Ev(ptr{{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @_ZN6MotherC2Ev(ptr {{.*}} %[[THIS]])
// OGCG:   %[[FATHER_BASE:.*]] = getelementptr{{.*}} i8, ptr %[[THIS]], i64 8
// OGCG:   call void @_ZN6FatherC2Ev(ptr{{.*}} %[[FATHER_BASE]])
// OGCG:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV5Child, i32 0, i32 0, i32 2), ptr %[[THIS]]
// OGCG:   %[[FATHER_BASE:.*]] = getelementptr{{.*}} i8, ptr %[[THIS]], i64 8
// OGCG:   store ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV5Child, i32 0, i32 1, i32 2), ptr %[[FATHER_BASE]]
// OGCG:   ret void
