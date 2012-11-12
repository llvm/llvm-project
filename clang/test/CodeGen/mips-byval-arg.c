// RUN: %clang -target mipsel-unknown-linux -O3 -S -o - -emit-llvm %s | FileCheck %s -check-prefix=O32
// RUN: %clang -target mips64el-unknown-linux -O3 -S -mabi=n64 -o - -emit-llvm %s | FileCheck %s -check-prefix=N64

typedef struct {
  float f[3];
} S0;

extern void foo2(S0);

// O32: define void @foo1(i32 %a0.coerce0, i32 %a0.coerce1, i32 %a0.coerce2)
// N64: define void @foo1(i64 %a0.coerce0, i32 %a0.coerce1)

void foo1(S0 a0) {
  foo2(a0);
}
