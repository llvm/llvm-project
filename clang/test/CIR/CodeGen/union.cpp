// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct { int x; } yolo;
typedef union { yolo y; struct { int lifecnt; }; } yolm;
typedef union { yolo y; struct { int *lifecnt; int genpad; }; } yolm2;

void m() {
  yolm q;
  yolm2 q2;
}

// CHECK: !ty_22struct2Eanon22 = !cir.struct<"struct.anon", !cir.ptr<i32>, i32, #cir.recdecl.ast>
// CHECK: !ty_22struct2Eyolo22 = !cir.struct<"struct.yolo", i32, #cir.recdecl.ast>
// CHECK: !ty_22union2Eyolm22 = !cir.struct<"union.yolm", !ty_22struct2Eyolo22>
// CHECK: !ty_22union2Eyolm222 = !cir.struct<"union.yolm2", !ty_22struct2Eanon22>

// CHECK:   cir.func @_Z1mv() {
// CHECK:   cir.alloca !ty_22union2Eyolm22, cir.ptr <!ty_22union2Eyolm22>, ["q"] {alignment = 4 : i64}
// CHECK:   cir.alloca !ty_22union2Eyolm222, cir.ptr <!ty_22union2Eyolm222>, ["q2"] {alignment = 8 : i64}