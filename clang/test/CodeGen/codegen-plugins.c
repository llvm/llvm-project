// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE
// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words | FileCheck %s --check-prefix=CHECK-ACTIVE
// RUN: %clang_cc1 -emit-llvm < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: not %clang_cc1 -emit-obj < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// REQUIRES: plugins, llvm-examples
// UNSUPPORTED: target={{.*windows.*}}
// Plugins are currently broken on AIX, at least in the CI.
// XFAIL: target={{.*}}-aix{{.*}}
// CHECK-INACTIVE-NOT: Bye
// CHECK-ACTIVE: CodeGen Bye
// CHECK-LLVM: define{{.*}} i32 @f
// CHECK-ERR: error: last words unsupported for binary output

int f(int x) {
  return x;
}
