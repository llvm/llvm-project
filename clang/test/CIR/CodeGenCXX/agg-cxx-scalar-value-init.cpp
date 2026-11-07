// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int a;
};

// CIR-LABEL: cir.func {{.*}} @_Z4testb
// OGCG-LABEL: define {{.*}} @_Z4testb
void test(bool b) {
  // CIR: cir.if %{{.+}} {
  // CIR:   %[[MEM:.*]] = cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
  // CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR:   cir.store align(4) %[[ZERO]], %[[MEM]] : !s32i, !cir.ptr<!s32i>
  // CIR: } else {
  // CIR:   %[[MEM2:.*]] = cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
  // CIR:   %[[ZERO2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR:   cir.store align(4) %[[ZERO2]], %[[MEM2]] : !s32i, !cir.ptr<!s32i>
  // CIR: }

  // OGCG: store i32 0, ptr %{{.*}}, align 4
  S s = b ? S{int()} : S{int()};
}
