// RUN: not %clang_cc1 -triple x86_64-pc-windows-gnu -Wno-unused-value -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=WIN64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=SYSV

struct S16 {
  long long a;
  long long b;
};

void take16(struct S16 s);

void give16(void) {
  struct S16 s = {1, 2};
  take16(s);
}

// Windows is not supported, so its signature is left as CIRGen emitted it.
// WIN64: declare void @take16(%struct.S16)
// SYSV:  declare void @take16(i64, i64)
