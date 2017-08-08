// Test that the always/never instrument lists apply.
// RUN: echo "fun:main" > %tmp-always.txt
// RUN: echo "fun:__xray*" > %tmp-never.txt
// RUN: %clangxx_xray \
// RUN:     -fxray-never-instrument=%tmp-never.txt \
// RUN:     -fxray-always-instrument=%tmp-always.txt \
// RUN:     %s -o %t
// RUN: %llvm_xray extract -symbolize %t | \
// RUN:    FileCheck %s --check-prefix NOINSTR
// RUN: %llvm_xray extract -symbolize %t | \
// RUN:    FileCheck %s --check-prefix ALWAYSINSTR
// REQUIRES: x86_64-linux
// REQUIRES: built-in-llvm-tree

// NOINSTR-NOT: {{.*__xray_NeverInstrumented.*}}
int __xray_NeverInstrumented() {
  return 0;
}

// ALWAYSINSTR: {{.*function-name:.*main.*}}
int main(int argc, char *argv[]) {
  return __xray_NeverInstrumented();
}
