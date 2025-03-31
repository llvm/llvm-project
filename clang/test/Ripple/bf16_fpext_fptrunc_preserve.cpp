// TODO: This remain until actual bf16 calculation support
// REQUIRES: x86-registered-target
// RUN: %clang %s -O2 -fenable-ripple -target x86_64 -mno-amx-bf16 -mno-avx512bf16 -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>
void add(__bf16 *c, __bf16 *a, __bf16 *b) {
    auto BS = ripple_set_block_shape(0, 64);
    size_t v = ripple_id(BS, 0);
    c[v] = a[v] + b[v];
// CHECK:        %[[LHSLoad:[0-9a-zA-Z_.]+]] = load <64 x bfloat>, ptr %a
// CHECK-NEXT:   %[[AddInLHS:[0-9a-zA-Z_.]+]] = fpext <64 x bfloat> %[[LHSLoad]] to <64 x float>
// CHECK-NEXT:   %[[RHSLoad:[0-9a-zA-Z_.]+]] = load <64 x bfloat>, ptr %b
// CHECK-NEXT:   %[[AddInRHS:[0-9a-zA-Z_.]+]] = fpext <64 x bfloat> %[[RHSLoad]] to <64 x float>
// CHECK-NEXT:   %[[AddOut:[0-9a-zA-Z_.]+]] = fadd <64 x float> %[[AddInLHS]], %[[AddInRHS]]
// CHECK-NEXT:   %{{.*}} = fptrunc <64 x float> %[[AddOut]] to <64 x bfloat>
}
