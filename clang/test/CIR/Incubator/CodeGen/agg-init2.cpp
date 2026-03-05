// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

// CHECK: !rec_Zero = !cir.record<struct "Zero" padded {!u8i}>

struct Zero {
  void yolo();
};

void f() {
  Zero z0 = Zero();
  // {} no element init.
  Zero z1 = Zero{};
}

// TODO: In this case, z1 gets "initialized" with an undef value. Should we
//       treat that as uninitialized? Should it even be happening?

// Trivial default constructor call is lowered away since it does nothing.
// CHECK: cir.func {{.*}} @_Z1fv()
// CHECK:     %[[Z0:.*]] = cir.alloca !rec_Zero, !cir.ptr<!rec_Zero>, ["z0", init]
// CHECK:     %[[Z1:.*]] = cir.alloca !rec_Zero, !cir.ptr<!rec_Zero>, ["z1", init]
// CHECK-NOT: cir.call @_ZN4ZeroC1Ev
// CHECK:     %[[UNDEF:.*]] = cir.const #cir.undef : !rec_Zero
// CHECK:     cir.store{{.*}} %[[UNDEF]], %[[Z1]] : !rec_Zero, !cir.ptr<!rec_Zero>
// CHECK:     cir.return

// LLVM-LABEL: define {{.*}} @_Z1fv()
// LLVM-NOT:     call {{.*}} @_ZN4ZeroC1Ev
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} @_Z1fv()
// OGCG-NOT:     call {{.*}} @_ZN4ZeroC1Ev
// OGCG:         ret void
