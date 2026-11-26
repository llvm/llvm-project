// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE
// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -mllvm -wave-goodbye 2>&1 | FileCheck %s --check-prefix=CHECK-ACTIVE
// RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -fpass-plugin-arg=Bye,-wave-goodbye 2>&1 | FileCheck %s --check-prefix=CHECK-ACTIVE
// RUN: not %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext -fpass-plugin-arg=Bye,-nope 2>&1 | FileCheck %s --check-prefix=BADOPT
// REQUIRES: plugins, llvm-examples
// UNSUPPORTED: target={{.*windows.*}}
// Plugins are currently broken on AIX, at least in the CI.
// XFAIL: target={{.*}}-aix{{.*}}
// CHECK-INACTIVE-NOT: Bye
// CHECK-ACTIVE: Bye: f
// BADOPT: error: Bye: Unknown command line argument '-nope'.

int f(int x) {
  return x;
}
