// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {stmt.return.flow.off} (P4317 A.1): flowing off the end of a
// value-returning function, when the caller uses the value, is undefined; under
// enforcement the function epilogue traps on the fall-through path.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z1fb
// CHECK: call void @llvm.ubsantrap(i8 11)
// OFF-LABEL: define {{.*}}@_Z1fb
// OFF-NOT: llvm.ubsantrap
int f(bool b) {
  if (b)
    return 1;
}
