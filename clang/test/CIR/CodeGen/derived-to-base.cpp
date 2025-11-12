// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// TODO(cir): The constructors in this test case are only here because we don't
//            have support for zero-initialization of base classes yet. We should
//            fix that soon.

struct Base {
  Base();
  void f();
  int a;
};

struct Derived : Base {
  Derived();
  double b;
};

void f() {
  Derived d;
  d.f();
}

// CIR: cir.func {{.*}} @_Z1fv()
// CIR:   %[[D:.*]] = cir.alloca !rec_Derived, !cir.ptr<!rec_Derived>, ["d", init]
// CIR:   cir.call @_ZN7DerivedC1Ev(%[[D]]) : (!cir.ptr<!rec_Derived>) -> ()
// CIR:   %[[D_BASE:.*]] = cir.base_class_addr %[[D]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   cir.call @_ZN4Base1fEv(%[[D_BASE]]) : (!cir.ptr<!rec_Base>) -> ()

// LLVM: define {{.*}}void @_Z1fv()
// LLVM:   %[[D:.*]] = alloca %struct.Derived
// LLVM:   call void @_ZN7DerivedC1Ev(ptr %[[D]])
// LLVM:   call void @_ZN4Base1fEv(ptr %[[D]])

// OGCG: define {{.*}}void @_Z1fv()
// OGCG:   %[[D:.*]] = alloca %struct.Derived
// OGCG:   call void @_ZN7DerivedC1Ev(ptr {{.*}} %[[D]])
// OGCG:   call void @_ZN4Base1fEv(ptr {{.*}} %[[D]])

void useBase(Base *base);
void callBaseUsingDerived(Derived *derived) {
  useBase(derived);
}


// CIR: cir.func {{.*}} @_Z20callBaseUsingDerivedP7Derived(%[[DERIVED_ARG:.*]]: !cir.ptr<!rec_Derived> {{.*}})
// CIR:   %[[DERIVED_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["derived", init]
// CIR:   cir.store %[[DERIVED_ARG]], %[[DERIVED_ADDR]]
// CIR:   %[[DERIVED:.*]] = cir.load{{.*}} %[[DERIVED_ADDR]]
// CIR:   %[[DERIVED_BASE:.*]] = cir.base_class_addr %[[DERIVED]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   cir.call @_Z7useBaseP4Base(%[[DERIVED_BASE]]) : (!cir.ptr<!rec_Base>) -> ()

// LLVM: define {{.*}} void @_Z20callBaseUsingDerivedP7Derived(ptr %[[DERIVED_ARG:.*]])
// LLVM:   %[[DERIVED_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[DERIVED_ARG]], ptr %[[DERIVED_ADDR]]
// LLVM:   %[[DERIVED:.*]] = load ptr, ptr %[[DERIVED_ADDR]]
// LLVM:   call void @_Z7useBaseP4Base(ptr %[[DERIVED]])

// OGCG: define {{.*}} void @_Z20callBaseUsingDerivedP7Derived(ptr {{.*}} %[[DERIVED_ARG:.*]])
// OGCG:   %[[DERIVED_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[DERIVED_ARG]], ptr %[[DERIVED_ADDR]]
// OGCG:   %[[DERIVED:.*]] = load ptr, ptr %[[DERIVED_ADDR]]
// OGCG:   call void @_Z7useBaseP4Base(ptr {{.*}} %[[DERIVED]])

Base *returnBaseFromDerived(Derived* derived) {
  return derived;
}

// CIR: cir.func {{.*}} @_Z21returnBaseFromDerivedP7Derived(%[[DERIVED_ARG:.*]]: !cir.ptr<!rec_Derived> {{.*}}) -> !cir.ptr<!rec_Base>
// CIR:   %[[DERIVED_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["derived", init]
// CIR:   %[[BASE_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Base>, !cir.ptr<!cir.ptr<!rec_Base>>, ["__retval"]
// CIR:   cir.store %[[DERIVED_ARG]], %[[DERIVED_ADDR]]
// CIR:   %[[DERIVED:.*]] = cir.load{{.*}} %[[DERIVED_ADDR]]
// CIR:   %[[DERIVED_BASE:.*]] = cir.base_class_addr %[[DERIVED]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   cir.store %[[DERIVED_BASE]], %[[BASE_ADDR]]
// CIR:   %[[BASE:.*]] = cir.load{{.*}} %[[BASE_ADDR]]
// CIR:   cir.return %[[BASE]] : !cir.ptr<!rec_Base>

// LLVM: define {{.*}} ptr @_Z21returnBaseFromDerivedP7Derived(ptr %[[DERIVED_ARG:.*]])
// LLVM:   %[[DERIVED_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[DERIVED_ARG]], ptr %[[DERIVED_ADDR]]
// LLVM:   %[[DERIVED:.*]] = load ptr, ptr %[[DERIVED_ADDR]]

// OGCG: define {{.*}} ptr @_Z21returnBaseFromDerivedP7Derived(ptr {{.*}} %[[DERIVED_ARG:.*]])
// OGCG:   %[[DERIVED_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[DERIVED_ARG]], ptr %[[DERIVED_ADDR]]
// OGCG:   %[[DERIVED:.*]] = load ptr, ptr %[[DERIVED_ADDR]]

volatile Derived derivedObj;

void test_volatile_store() {
  derivedObj.a = 0;
}

// CIR: cir.func {{.*}} @_Z19test_volatile_storev()
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[DERIVED_OBJ:.*]] = cir.get_global @derivedObj : !cir.ptr<!rec_Derived>
// CIR:   %[[DERIVED_OBJ_BASE:.*]] = cir.base_class_addr %[[DERIVED_OBJ]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[DERIVED_OBJ_A:.*]] = cir.get_member %[[DERIVED_OBJ_BASE]][0] {name = "a"} : !cir.ptr<!rec_Base> -> !cir.ptr<!s32i>
// CIR:   cir.store volatile {{.*}} %[[ZERO]], %[[DERIVED_OBJ_A]] : !s32i, !cir.ptr<!s32i>

// LLVM: define {{.*}} void @_Z19test_volatile_storev()
// LLVM:   store volatile i32 0, ptr @derivedObj

// OGCG: define {{.*}} void @_Z19test_volatile_storev()
// OGCG:   store volatile i32 0, ptr @derivedObj

void test_volatile_load() {
  [[maybe_unused]] int val = derivedObj.a;
}

// CIR: cir.func {{.*}} @_Z18test_volatile_loadv()
// CIR:   %[[DERIVED_OBJ:.*]] = cir.get_global @derivedObj : !cir.ptr<!rec_Derived>
// CIR:   %[[DERIVED_OBJ_BASE:.*]] = cir.base_class_addr %[[DERIVED_OBJ]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   %[[DERIVED_OBJ_A:.*]] = cir.get_member %[[DERIVED_OBJ_BASE]][0] {name = "a"} : !cir.ptr<!rec_Base> -> !cir.ptr<!s32i>
// CIR:   %[[VAL:.*]] = cir.load volatile {{.*}} %[[DERIVED_OBJ_A]] : !cir.ptr<!s32i>, !s32i

// LLVM: define {{.*}} void @_Z18test_volatile_loadv()
// LLVM:   %[[VAL_ADDR:.*]] = alloca i32
// LLVM:   %[[DERIVED_OBJ:.*]] = load volatile i32, ptr @derivedObj

// OGCG: define {{.*}} void @_Z18test_volatile_loadv()
// OGCG:   %[[VAL_ADDR:.*]] = alloca i32
// OGCG:   %[[DERIVED_OBJ:.*]] = load volatile i32, ptr @derivedObj
// OGCG:   store i32 %[[DERIVED_OBJ]], ptr %[[VAL_ADDR]]
