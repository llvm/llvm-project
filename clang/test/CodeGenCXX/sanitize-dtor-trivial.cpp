// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

// TODO Success pending on resolution of issue:
//    https://github.com/google/sanitizers/issues/596
// XFAIL: *

struct Trivial {
  int a;
  int b;
};
Trivial t;

// CHECK: call void @__sanitizer_dtor_callback({{.*}}, !dbg ![[DI:[0-9]+]]


// CHECK-LABEL: !DIFile{{.*}}cpp

// CHECK-DAG: ![[DI]] = {{.*}}line: [[@LINE-28]]
