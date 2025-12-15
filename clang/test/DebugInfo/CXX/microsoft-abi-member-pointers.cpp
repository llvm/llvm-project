// RUN: %clang_cc1 -std=c++11 -debug-info-kind=line-tables-only -fno-rtti -emit-llvm %s -o - -triple=x86_64-pc-win32 -fms-extensions | FileCheck %s

struct Task {
  virtual void Run() = 0;
};

auto b = &Task::Run;

// CHECK: define {{.*}}@"??_9Task@@$BA@AA"
// CHECK-NOT: define
// CHECK: musttail call {{.*}}, !dbg ![[DBG:[0-9]+]]

// CHECK: ![[DBG]] = !DILocation(line: 4

