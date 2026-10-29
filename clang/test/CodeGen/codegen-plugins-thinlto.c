// REQUIRES: plugins, llvm-examples
// UNSUPPORTED: target={{.*windows.*}}
// Plugins are currently broken on AIX, at least in the CI.
// XFAIL: target={{.*}}-aix{{.*}}

// RUN: %clang_cc1 -O2 -flto=thin -emit-llvm-bc %s -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -S -o - -fpass-plugin=%llvmshlibdir/Bye%pluginext 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -S -o - -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words | FileCheck %s --check-prefix=CHECK-ACTIVE
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -emit-llvm -o - -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: not %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -emit-obj -o - -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -last-words 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

// CHECK-INACTIVE-NOT: Bye
// CHECK-INACTIVE: f:
// CHECK-ACTIVE: CodeGen Bye
// CHECK-LLVM: define{{.*}} i32 @f
// CHECK-ERR: error: last words unsupported for binary output

int f(int x) {
  return x;
}
