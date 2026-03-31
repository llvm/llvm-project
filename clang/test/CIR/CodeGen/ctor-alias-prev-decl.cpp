// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test that constructor aliases work correctly when the call site is emitted
// before the constructor definition. This creates a declaration for the
// complete constructor first, which is then replaced by the alias.

struct B {
  B();
};

void bar() {
  B b;
}

B::B() {
}

// CHECK: cir.func{{.*}} @_Z3barv()
// CHECK:   %[[B:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["b", init]
// CHECK:   cir.call @_ZN1BC1Ev(%[[B]]) : (!cir.ptr<!rec_B> {{.*}}) -> ()
// CHECK:   cir.return

// CHECK: cir.func{{.*}} @_ZN1BC2Ev(%arg0: !cir.ptr<!rec_B>
// CHECK:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["this", init]
// CHECK:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>

// CHECK: cir.func{{.*}} private dso_local @_ZN1BC1Ev(!cir.ptr<!rec_B>) alias(@_ZN1BC2Ev)

// OGCG: @_ZN1BC1Ev = unnamed_addr alias void (ptr), ptr @_ZN1BC2Ev

// OGCG: define{{.*}} void @_Z3barv()
// OGCG:   %[[B:.*]] = alloca %struct.B, align 1
// OGCG:   call void @_ZN1BC1Ev(ptr{{.*}} %[[B]])
// OGCG:   ret void

// OGCG: define{{.*}} @_ZN1BC2Ev(ptr{{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
