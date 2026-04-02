// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test that destructor aliases work correctly when the call site is emitted
// before the destructor definition. This creates a declaration for the
// complete destructor first, which is then replaced by the alias.

struct B {
  ~B();
};

void baz() {
  B b;
}

B::~B() {
}

// CHECK: cir.func{{.*}} @_Z3bazv()
// CHECK:   %[[B:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["b"]
// CHECK:   cir.call @_ZN1BD1Ev(%[[B]]) nothrow : (!cir.ptr<!rec_B> {{.*}}) -> ()

// CHECK: cir.func{{.*}} @_ZN1BD2Ev(%arg0: !cir.ptr<!rec_B>
// CHECK:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["this", init]
// CHECK:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>

// CHECK: cir.func{{.*}} private dso_local @_ZN1BD1Ev(!cir.ptr<!rec_B>) alias(@_ZN1BD2Ev)

// OGCG: @_ZN1BD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1BD2Ev

// OGCG: define{{.*}} void @_Z3bazv()
// OGCG:   %[[B:.*]] = alloca %struct.B, align 1
// OGCG:   call void @_ZN1BD1Ev(ptr{{.*}} %[[B]])
// OGCG:   ret void

// OGCG: define{{.*}} @_ZN1BD2Ev(ptr{{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
