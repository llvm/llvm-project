// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

struct Base1 {
  virtual ~Base1();
};

struct Base2 {
  virtual ~Base2();
};

struct Derived final : Base1 {};

Derived *ptr_cast(Base1 *ptr) {
  return dynamic_cast<Derived *>(ptr);
  //      CHECK: %[[#SRC:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_Base1_>>, !cir.ptr<!ty_Base1_>
  // CHECK-NEXT: %[[#EXPECTED_VPTR:]] = cir.vtable.address_point(@_ZTV7Derived, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
  // CHECK-NEXT: %[[#SRC_VPTR_PTR:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_Base1_>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
  // CHECK-NEXT: %[[#SRC_VPTR:]] = cir.load %[[#SRC_VPTR_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>, !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
  // CHECK-NEXT: %[[#SUCCESS:]] = cir.cmp(eq, %[[#SRC_VPTR]], %[[#EXPECTED_VPTR]]) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.ternary(%[[#SUCCESS]], true {
  // CHECK-NEXT:   %[[#RES:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_Base1_>), !cir.ptr<!ty_Derived>
  // CHECK-NEXT:   cir.yield %[[#RES]] : !cir.ptr<!ty_Derived>
  // CHECK-NEXT: }, false {
  // CHECK-NEXT:   %[[#NULL:]] = cir.const #cir.ptr<null> : !cir.ptr<!ty_Derived>
  // CHECK-NEXT:   cir.yield %[[#NULL]] : !cir.ptr<!ty_Derived>
  // CHECK-NEXT: }) : (!cir.bool) -> !cir.ptr<!ty_Derived>
}

//      LLVM: define dso_local ptr @_Z8ptr_castP5Base1(ptr readonly %[[#SRC:]])
// LLVM-NEXT:   %[[#VPTR:]] = load ptr, ptr %[[#SRC]], align 8
// LLVM-NEXT:   %[[#SUCCESS:]] = icmp eq ptr %[[#VPTR]], getelementptr inbounds (i8, ptr @_ZTV7Derived, i64 16)
// LLVM-NEXT:   %[[RESULT:.+]] = select i1 %[[#SUCCESS]], ptr %[[#SRC]], ptr null
// LLVM-NEXT:   ret ptr %[[RESULT]]
// LLVM-NEXT: }

Derived &ref_cast(Base1 &ref) {
  return dynamic_cast<Derived &>(ref);
  //      CHECK: %[[#SRC:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_Base1_>>, !cir.ptr<!ty_Base1_>
  // CHECK-NEXT: %[[#EXPECTED_VPTR:]] = cir.vtable.address_point(@_ZTV7Derived, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
  // CHECK-NEXT: %[[#SRC_VPTR_PTR:]] = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_Base1_>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
  // CHECK-NEXT: %[[#SRC_VPTR:]] = cir.load %[[#SRC_VPTR_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>, !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
  // CHECK-NEXT: %[[#SUCCESS:]] = cir.cmp(eq, %[[#SRC_VPTR]], %[[#EXPECTED_VPTR]]) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.bool
  // CHECK-NEXT: %[[#FAILED:]] = cir.unary(not, %[[#SUCCESS]]) : !cir.bool, !cir.bool
  // CHECK-NEXT: cir.if %[[#FAILED]] {
  // CHECK-NEXT:   cir.call @__cxa_bad_cast() : () -> ()
  // CHECK-NEXT:   cir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: %{{.+}} = cir.cast(bitcast, %[[#SRC]] : !cir.ptr<!ty_Base1_>), !cir.ptr<!ty_Derived>
}

//      LLVM: define dso_local noundef ptr @_Z8ref_castR5Base1(ptr readonly returned %[[#SRC:]])
// LLVM-NEXT:   %[[#VPTR:]] = load ptr, ptr %[[#SRC]], align 8
// LLVM-NEXT:   %[[OK:.+]] = icmp eq ptr %[[#VPTR]], getelementptr inbounds (i8, ptr @_ZTV7Derived, i64 16)
// LLVM-NEXT:   br i1 %[[OK]], label %[[#LABEL_OK:]], label %[[#LABEL_FAIL:]]
//      LLVM: [[#LABEL_FAIL]]:
// LLVM-NEXT:   tail call void @__cxa_bad_cast()
// LLVM-NEXT:   unreachable
//      LLVM: [[#LABEL_OK]]:
// LLVM-NEXT:   ret ptr %[[#SRC]]
// LLVM-NEXT: }

Derived *ptr_cast_always_fail(Base2 *ptr) {
  return dynamic_cast<Derived *>(ptr);
  //      CHECK: %{{.+}} = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_Base2_>>, !cir.ptr<!ty_Base2_>
  // CHECK-NEXT: %[[#RESULT:]] = cir.const #cir.ptr<null> : !cir.ptr<!ty_Derived>
  // CHECK-NEXT: cir.store %[[#RESULT]], %{{.+}} : !cir.ptr<!ty_Derived>, !cir.ptr<!cir.ptr<!ty_Derived>>
}

//      LLVM: define dso_local noalias noundef ptr @_Z20ptr_cast_always_failP5Base2(ptr nocapture readnone %{{.+}})
// LLVM-NEXT:   ret ptr null
// LLVM-NEXT: }

Derived &ref_cast_always_fail(Base2 &ref) {
  return dynamic_cast<Derived &>(ref);
  //      CHECK: %{{.+}} = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_Base2_>>, !cir.ptr<!ty_Base2_>
  // CHECK-NEXT: %{{.+}} = cir.const #cir.ptr<null> : !cir.ptr<!ty_Derived>
  // CHECK-NEXT: cir.call @__cxa_bad_cast() : () -> ()
  // CHECK-NEXT: cir.unreachable
}

//      LLVM: define dso_local noalias noundef nonnull ptr @_Z20ref_cast_always_failR5Base2(ptr nocapture readnone %{{.+}})
// LLVM-NEXT:   tail call void @__cxa_bad_cast()
// LLVM-NEXT:   unreachable
// LLVM-NEXT: }
