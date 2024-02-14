// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

static const int g = 1;
void foo() {
  if ((g != 1) && (g != 1))
    return;
  if ((g == 1) || (g == 1))
    return;
}
// CHECK:  cir.func no_proto @foo()
// CHECK:    cir.scope {
// CHECK:      [[ZERO:%.*]] = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:      [[FALSE:%.*]] = cir.cast(int_to_bool, [[ZERO:%.*]] : !s32i), !cir.bool
// CHECK:      cir.if [[FALSE]] {
// CHECK:        cir.return
// CHECK:      }
// CHECK:    }
// CHECK:    cir.return

typedef struct { int x; } S;
static const S s = {0};
void bar() {
  int a =  s.x;
}
// CHECK:  cir.func no_proto @bar()
// CHECK:    [[ALLOC:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK:    {{%.*}} = cir.get_global @s : cir.ptr <!ty_22S22>
// CHECK:    [[CONST:%.*]] = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:    cir.store [[CONST]], [[ALLOC]] : !s32i, cir.ptr <!s32i>
// CHECK:    cir.return

