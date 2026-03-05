// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {
  int a;
  int b;
} __attribute__((alligned (4))) S;

// CHECK: cir.func {{.*@init}}() -> !u64i
// CHECK:    %[[#V0:]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[#V1:]] = cir.cast bitcast %[[#V0]] : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CHECK:    %[[#V2:]] = cir.load %[[#V1]] : !cir.ptr<!u64i>, !u64i
// CHECK:    cir.return %[[#V2]] : !u64i
S init() {
  S s;
  return s;
}
