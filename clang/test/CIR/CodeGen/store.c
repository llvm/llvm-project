// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo(void) {
  int a = 0;
  a = 1;
}

//      CHECK: cir.func @foo()
// CHECK-NEXT:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   cir.store %1, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   %2 = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:   cir.store %2, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

typedef int (*fn_t)();
int get42() { return 42; }

void storeNoArgsFn() {
  fn_t f = get42;
}

// CHECK:  cir.func {{.*@storeNoArgsFn}}
// CHECK:    %0 = cir.alloca
// CHECK:    %1 = cir.get_global @get42 : !cir.ptr<!cir.func<() -> !s32i>>
// CHECK:    %2 = cir.cast(bitcast, %1 : !cir.ptr<!cir.func<() -> !s32i>>), !cir.ptr<!cir.func<(...) -> !s32i>>
// CHECK:    cir.store %2, %0 : !cir.ptr<!cir.func<(...) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(...) -> !s32i>>>

