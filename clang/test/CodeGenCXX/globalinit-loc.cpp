// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// rdar://problem/14985269.
//
// Verify that the global init helper function does not get associated
// with any source location.
//
// CHECK: define internal void @_GLOBAL__sub_I_globalinit_loc.cpp
// CHECK: !dbg ![[DBG:.*]]
// CHECK:  !"0x2e\00\00\00_GLOBAL__sub_I_globalinit_loc.cpp\000\00{{.*}}\000", {{.*}} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
// CHECK: ![[DBG]] = !MDLocation(line: 0,
# 99 "someheader.h"
class A {
public:
  A();
  int foo() { return 0; }
};
# 5 "main.cpp"
A a;

int f() {
  return a.foo();
}

