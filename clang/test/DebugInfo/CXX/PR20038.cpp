// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -mllvm -no-discriminators -emit-llvm  %s -o - | FileCheck %s

struct C {
  ~C();
};
extern bool b;
// CHECK: call {{.*}}, !dbg [[DTOR_CALL1_LOC:![0-9]*]]
// CHECK: call {{.*}}, !dbg [[DTOR_CALL2_LOC:![0-9]*]]
// CHECK: [[FUN1:.*]] = distinct !DISubprogram(name: "fun1",{{.*}} DISPFlagDefinition
// CHECK: [[DTOR_CALL1_LOC]] = !DILocation(line: [[@LINE+1]], column: 15, scope: [[FUN1]])
void fun1() { b && (C(), 1); }
// CHECK: [[FUN2:.*]] = distinct !DISubprogram(name: "fun2",{{.*}} DISPFlagDefinition
// CHECK: [[DTOR_CALL2_LOC]] = !DILocation(line: [[@LINE+1]], column: 15, scope: [[FUN2]])
bool fun2() { return (C(), b) && 0; }
