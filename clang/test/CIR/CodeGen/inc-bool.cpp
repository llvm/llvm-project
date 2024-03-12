// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo(bool x) {
  x++;
}

// CHECK:  cir.func @_Z3foob(%arg0: !cir.bool loc({{.*}}))
// CHECK:    [[ALLOC_X:%.*]] = cir.alloca !cir.bool, cir.ptr <!cir.bool>, ["x", init] {alignment = 1 : i64}
// CHECK:    cir.store %arg0, [[ALLOC_X]] : !cir.bool, cir.ptr <!cir.bool>
// CHECK:    {{.*}} = cir.load [[ALLOC_X]] : cir.ptr <!cir.bool>, !cir.bool
// CHECK:    [[TRUE:%.*]] = cir.const(#true) : !cir.bool
// CHECK:    cir.store [[TRUE]], [[ALLOC_X]] : !cir.bool, cir.ptr <!cir.bool>
// CHECK:    cir.return
