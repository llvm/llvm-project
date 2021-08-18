// Without hotness threshold, print both hot and cold remarks.
// RUN: %clang_cc1 -triple x86_64-linux %s -emit-llvm-only -O3 \
// RUN:     -fprofile-sample-use=%S/Inputs/remarks-hotness.prof \
// RUN:     -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline \
// RUN:     -fexperimental-new-pass-manager -fdiagnostics-show-hotness 2>&1 \
// RUN:     | FileCheck -check-prefix=REMARKS %s

// With auto hotness threshold, only print hot remarks.
// RUN: %clang_cc1 -triple x86_64-linux %s -emit-llvm-only -O3 \
// RUN:     -fprofile-sample-use=%S/Inputs/remarks-hotness.prof \
// RUN:     -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline \
// RUN:     -fexperimental-new-pass-manager -fdiagnostics-show-hotness \
// RUN:     -fdiagnostics-hotness-threshold=auto 2>&1 \
// RUN:     | FileCheck -check-prefix=HOT_CALL %s

int callee1() {
  return 1;
}

__attribute__((noinline)) int callee2() {
  return 2;
}

// REMARKS: '_Z7callee1v' inlined into '_Z7caller1v'
// HOT_CALL: '_Z7callee1v' inlined into '_Z7caller1v'
int caller1() {
  return callee1();
}

// REMARKS: '_Z7callee2v' not inlined into '_Z7caller2v'
// HOT_CALL-NOT: '_Z7callee2v' not inlined into '_Z7caller2v'
int caller2() {
  return callee2();
}
