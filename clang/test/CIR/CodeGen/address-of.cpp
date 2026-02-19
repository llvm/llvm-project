// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Container {
  int x;
  int y;
};

void builtin_address_of() {
  Container a;
  Container* b = __builtin_addressof(a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_Container, !cir.ptr<!rec_Container>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Container>, !cir.ptr<!cir.ptr<!rec_Container>>, ["b", init]
// CIR: cir.store {{.*}} %[[A_ADDR]], %[[B_ADDR]] : !cir.ptr<!rec_Container>, !cir.ptr<!cir.ptr<!rec_Container>>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.Container, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr %[[A_ADDR]], ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca %struct.Container, align 4
// OGCG: %[[B_ADDR:.*]] = alloca ptr, align 8
// OGCG: store ptr %[[A_ADDR]], ptr %[[B_ADDR]], align 8
