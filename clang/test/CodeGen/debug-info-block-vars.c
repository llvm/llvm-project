// RUN: %clang_cc1 -x c -std=c89 -fblocks -debug-info-kind=standalone -emit-llvm -O0 \
// RUN:   -triple x86_64-apple-darwin -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c -std=c89 -fblocks -debug-info-kind=standalone -emit-llvm -O1 \
// RUN:   -triple x86_64-apple-darwin -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-OPT %s

// CHECK: define internal void @__f_block_invoke(ptr noundef %.block_descriptor)
// CHECK: %.block_descriptor.addr = alloca ptr, align 8
// CHECK: %block.addr = alloca ptr, align 8
// CHECK: store ptr %.block_descriptor, ptr %.block_descriptor.addr, align 8
// CHECK: call void @llvm.dbg.declare(metadata ptr %.block_descriptor.addr,
// CHECK-SAME:                        metadata !DIExpression())
// CHECK-OPT-NOT: alloca
// Since the block address is not used anywhere in this function,
// the optimizer (DeadArgElim) has replaced all the false uses
// (i.e., metadata users) with poison.
// CHECK-OPT: call void @llvm.dbg.value(metadata ptr poison,
// CHECK-OPT-SAME:                      metadata !DIExpression())
void f(void) {
  a(^{
    b();
  });
}
