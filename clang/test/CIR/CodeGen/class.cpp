// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// CIR: !rec_IncompleteC = !cir.record<class "IncompleteC" incomplete>
// CIR: !rec_Base = !cir.record<class "Base" {!s32i}>
// CIR: !rec_CompleteC = !cir.record<class "CompleteC" {!s32i, !s8i}>
// CIR: !rec_Derived = !cir.record<class "Derived" {!rec_Base, !s32i}>

// Note: LLVM and OGCG do not emit the type for incomplete classes.

// LLVM: %class.CompleteC = type { i32, i8 }
// LLVM: %class.Derived = type { %class.Base, i32 }
// LLVM: %class.Base = type { i32 }

// OGCG: %class.CompleteC = type { i32, i8 }
// OGCG: %class.Derived = type { %class.Base, i32 }
// OGCG: %class.Base = type { i32 }

class IncompleteC;
IncompleteC *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteC>
// LLVM: @p = global ptr null
// OGCG: @p = global ptr null, align 8

class CompleteC {
public:    
  int a;
  char b;
};

CompleteC cc;

// CIR:       cir.global external @cc = #cir.zero : !rec_CompleteC
// LLVM:  @cc = global %class.CompleteC zeroinitializer
// OGCG:  @cc = global %class.CompleteC zeroinitializer

class Base {
public:
  int a;
};

class Derived : public Base {
public:
  int b;
};

int use(Derived *d) { return d->b; }

// CIR: cir.func{{.*}} @_Z3useP7Derived(%[[ARG0:.*]]: !cir.ptr<!rec_Derived>
// CIR:  %[[D_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["d", init]
// CIR:  cir.store %[[ARG0]], %[[D_ADDR]]
// CIR:  %[[D_PTR:.*]] = cir.load align(8) %0
// CIR:  %[[D_B_ADDR:.*]] = cir.get_member %[[D_PTR]][1] {name = "b"}
// CIR:  %[[D_B:.*]] = cir.load align(4) %[[D_B_ADDR]]

// LLVM: define{{.*}} i32 @_Z3useP7Derived
// LLVM:   getelementptr %class.Derived, ptr %{{.*}}, i32 0, i32 1

// OGCG: define{{.*}} i32 @_Z3useP7Derived
// OGCG:   getelementptr inbounds nuw %class.Derived, ptr %{{.*}}, i32 0, i32 1

int use_base() {
  Derived d;
  return d.a;
}

// CIR: cir.func{{.*}} @_Z8use_basev
// CIR:   %[[D_ADDR:.*]] = cir.alloca !rec_Derived, !cir.ptr<!rec_Derived>, ["d"]
// CIR:   %[[BASE_ADDR:.*]] cir.base_class_addr %[[D_ADDR]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[D_A_ADDR:.*]] = cir.get_member %2[0] {name = "a"} : !cir.ptr<!rec_Base> -> !cir.ptr<!s32i>
// CIR:   %[[D_A:.*]] = cir.load align(4) %3 : !cir.ptr<!s32i>, !s32i

// LLVM: define{{.*}} i32 @_Z8use_basev
// LLVM:   %[[D:.*]] = alloca %class.Derived
// LLVM:   %[[D_A_ADDR:.*]] = getelementptr %class.Base, ptr %[[D]], i32 0, i32 0

// OGCG: define{{.*}} i32 @_Z8use_basev
// OGCG:   %[[D:.*]] = alloca %class.Derived
// OGCG:   %[[D_A_ADDR:.*]] = getelementptr inbounds nuw %class.Base, ptr %[[D]], i32 0, i32 0

int use_base_via_pointer(Derived *d) {
  return d->a;
}

// CIR: cir.func{{.*}} @_Z20use_base_via_pointerP7Derived(%[[ARG0:.*]]: !cir.ptr<!rec_Derived>
// CIR:   %[[D_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["d", init]
// CIR:   cir.store %[[ARG0]], %[[D_ADDR]]
// CIR:   %[[D:.*]] = cir.load align(8) %[[D_ADDR]]
// CIR:   %[[BASE_ADDR:.*]] = cir.base_class_addr %[[D]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[D_A_ADDR:.*]] = cir.get_member %[[BASE_ADDR]][0] {name = "a"}
// CIR:   %[[D_A:.*]] = cir.load align(4) %[[D_A_ADDR]]

// LLVM: define{{.*}} i32 @_Z20use_base_via_pointerP7Derived
// LLVM:   %[[D_A_ADDR:.*]] = getelementptr %class.Base, ptr %{{.*}}, i32 0, i32 0

// OGCG: define{{.*}} i32 @_Z20use_base_via_pointerP7Derived
// OGCG:   %[[D_A_ADDR:.*]] = getelementptr inbounds nuw %class.Base, ptr %{{.*}}, i32 0, i32 0

struct EmptyDerived : Base {};
struct EmptyDerived2 : EmptyDerived {};

void use_empty_derived2() {
  EmptyDerived2 d2;
}

// CIR: cir.func{{.*}} @_Z18use_empty_derived2v()
// CIR:   %0 = cir.alloca !rec_EmptyDerived2, !cir.ptr<!rec_EmptyDerived2>, ["d2"]
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z18use_empty_derived2v
// LLVM:   alloca %struct.EmptyDerived2
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z18use_empty_derived2v
// OGCG:   alloca %struct.EmptyDerived2
// OGCG:   ret void
