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

