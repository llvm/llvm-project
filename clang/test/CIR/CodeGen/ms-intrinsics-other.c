// RUN: %clang_cc1 -ffreestanding -fms-extensions -Wno-implicit-function-declaration -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -ffreestanding -fms-extensions -Wno-implicit-function-declaration -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

// This test mimics clang/test/CodeGen/ms-intrinsics-other.c, which eventually
// CIR shall be able to support fully.

unsigned short test__lzcnt16(unsigned short x) {
  return __lzcnt16(x);
}
// CIR-LABEL: test__lzcnt16
// CIR: {{%.*}} = cir.clz {{%.*}} : !u16i
// LLVM-LABEL: test__lzcnt16
// LLVM: {{%.*}} = call i16 @llvm.ctlz.i16(i16 {{%.*}}, i1 false)

unsigned int test__lzcnt(unsigned int x) {
  return __lzcnt(x);
}
// CIR-LABEL: test__lzcnt
// CIR: {{%.*}} = cir.clz {{%.*}} : !u32i
// LLVM-LABEL: test__lzcnt
// LLVM:  {{%.*}} = call i32 @llvm.ctlz.i32(i32 {{%.*}}, i1 false)

unsigned __int64 test__lzcnt64(unsigned __int64 x) {
  return __lzcnt64(x);
}
// CIR-LABEL: test__lzcnt64
// CIR: {{%.*}} = cir.clz {{%.*}} : !u64i
// LLVM-LABEL: test__lzcnt64
// LLVM: {{%.*}} = call i64 @llvm.ctlz.i64(i64 {{%.*}}, i1 false)

unsigned short test__popcnt16(unsigned short x) {
  return __popcnt16(x);
}
// CIR-LABEL: test__popcnt16
// CIR: {{%.*}} = cir.popcount {{%.*}} : !u16i
// LLVM-LABEL: test__popcnt16
// LLVM: {{%.*}} = call i16 @llvm.ctpop.i16(i16 {{%.*}})

unsigned int test__popcnt(unsigned int x) {
  return __popcnt(x);
}
// CIR-LABEL: test__popcnt
// CIR: {{%.*}} = cir.popcount {{%.*}} : !u32i
// LLVM-LABEL: test__popcnt
// LLVM: {{%.*}} = call i32 @llvm.ctpop.i32(i32 {{%.*}})

unsigned __int64 test__popcnt64(unsigned __int64 x) {
  return __popcnt64(x);
}
// CIR-LABEL: test__popcnt64
// CIR: {{%.*}} = cir.popcount {{%.*}} : !u64i
// LLVM-LABEL: test__popcnt64
// LLVM: {{%.*}} = call i64 @llvm.ctpop.i64(i64 {{%.*}})
