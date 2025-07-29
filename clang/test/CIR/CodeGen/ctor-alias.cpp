// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct B {
  B();
};
B::B() {
}

// OGCG: @_ZN1BC1Ev = unnamed_addr alias void (ptr), ptr @_ZN1BC2Ev

// CHECK: cir.func{{.*}} @_ZN1BC2Ev(%arg0: !cir.ptr<!rec_B>
// CHECK:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["this", init]
// CHECK:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>

// CHECK: cir.func{{.*}} private dso_local @_ZN1BC1Ev(!cir.ptr<!rec_B>) alias(@_ZN1BC2Ev)

// LLVM: define{{.*}} @_ZN1BC2Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]

// This should be an alias, like the similar OGCG alias above, but that's not
// implemented yet.
// LLVM: declare dso_local void @_ZN1BC1Ev(ptr)

// OGCG: define{{.*}} @_ZN1BC2Ev(ptr{{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]

// The constructor in this cases is handled by RAUW rather than aliasing.
struct Struk {
  Struk() {}
};

void baz() {
  Struk s;
}

// CHECK:   cir.func{{.*}} @_ZN5StrukC2Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init]
// CHECK:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK:     cir.return

// CHECK-NOT:   cir.func{{.*}} @_ZN5StrukC1Ev

// CHECK:   cir.func{{.*}} @_Z3bazv()
// CHECK:     %[[S_ADDR:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s", init]
// CHECK:     cir.call @_ZN5StrukC2Ev(%[[S_ADDR]]) : (!cir.ptr<!rec_Struk>) -> ()

// LLVM: define linkonce_odr void @_ZN5StrukC2Ev(ptr %[[THIS_ARG]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]

// LLVM: define{{.*}} void @_Z3bazv()
// LLVM:   %[[S_ADDR:.*]] = alloca %struct.Struk
// LLVM:   call void @_ZN5StrukC2Ev(ptr{{.*}} %[[S_ADDR]])

// This function gets emitted before the constructor in OGCG.
// OGCG: define{{.*}} void @_Z3bazv()
// OGCG:   %[[S_ADDR:.*]] = alloca %struct.Struk
// OGCG:   call void @_ZN5StrukC2Ev(ptr{{.*}} %[[S_ADDR]])

// OGCG: define linkonce_odr void @_ZN5StrukC2Ev(ptr{{.*}} %[[THIS_ARG]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
