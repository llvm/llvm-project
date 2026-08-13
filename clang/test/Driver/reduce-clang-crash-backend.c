// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %s %t/test.c

// RUN: echo "%clang -target x86_64-unknown-linux-gnu -O2 -mllvm -codegen-pipeline-trigger-crash -mllvm=-enable-scalar-pre -c %t/test.c" > %t/crash.sh
// Speed up the test by only reducing functions in llvm-reduce.
// RUN: %python %S/../../utils/reduce-clang-crash.py %t/crash.sh %t/test.c --clang %clang --llvm-reduce-flag=--delta-passes=functions -v

// RUN: FileCheck --check-prefix=CHECK-IR %s < %t/test.reduced.ll
// RUN: FileCheck --check-prefix=CHECK-CMD %s < %t/crash.reduced.sh

// CHECK-IR: define {{.*}}@
// CHECK-IR-NOT: define

// CHECK-CMD: {{^.*}}llc{{(\.exe)?}} -codegen-pipeline-trigger-crash {{[^ ]*}}test.reduced.ll{{$}}

void foo() {}
void bar() {}
