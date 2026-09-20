// RUN: not %clang_cc1 %s -triple mipsel-unknown-linux-gnu -emit-llvm -o /dev/null 2>&1 \
// RUN:   | FileCheck %s
// RUN: not %clang_cc1 %s -triple mipsel-unknown-linux-gnu -target-feature +mips16 \
// RUN:   -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-MIPS16

// CHECK: error: 'musttail' attribute for this call is impossible because calls outside the current linkage unit cannot be tail called on MIPS
// CHECK-MIPS16: error: 'musttail' attribute for this call is impossible because the MIPS16 ABI does not support tail calls

static int local(int x) { return x; }

int call_local(int x) {
  [[clang::musttail]] return local(x);
}

extern int external(int x);

int call_external(int x) {
  [[clang::musttail]] return external(x);
}
