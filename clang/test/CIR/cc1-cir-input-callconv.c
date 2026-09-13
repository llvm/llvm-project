// Verify the .cir cc1 input path runs the full CIR-to-CIR pipeline, including
// CallConvLowering. We first emit *un-lowered* CIR (-clangir-disable-passes)
// so the serialized module still returns the struct by value; feeding it back
// through the .cir path must apply the x86_64 System V ABI, coercing the
// 8-byte struct return to i64. Before the .cir path shared runCIRToCIRPasses
// it skipped CallConvLowering and this coercion would not happen.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -clangir-disable-passes -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=LLVM

struct S {
  int a, b;
};

struct S make(int x) {
  struct S s = {x, x};
  return s;
}

// The serialized module is still high-level: the struct is returned by value,
// not yet coerced to an ABI type.
// CIR: cir.func {{.*}}@make(%arg0: !s32i {{.*}}) -> !rec_S

// After the .cir path runs CallConvLowering, the 8-byte struct return is
// coerced to i64 per the x86_64 System V ABI.
// LLVM: define {{.*}} i64 @make(i32 {{.*}}%{{.+}})
