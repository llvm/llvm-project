// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O3 \
// RUN:   -debug-info-kind=limited -o - -debugger-tuning=gdb -dwarf-version=4 \
// RUN:   | FileCheck %s

// Makes sure there is no !dbg between function attributes and '{'.
// CHECK-LABEL: define void @foo{{.*}} #{{[0-9]+}} {
// CHECK-NOT: ret {{.*}}!dbg
__attribute__((nodebug)) void foo(int *a) {
  *a = 1;
}

// CHECK-LABEL: define {{.*}}@bar{{.*}}!dbg
void bar(int *a) {
  foo(a);
}
