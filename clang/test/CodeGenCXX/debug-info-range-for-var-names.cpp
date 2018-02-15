// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

struct vec {
  using itr = int*;
  itr begin() { return nullptr; }
  itr end() { return nullptr; }
};

void test() {
  vec as, bs, cs;

  for (auto a : as)
    for (auto b : bs)
      for (auto c : cs) {
      }
}

// CHECK: call void @llvm.dbg.declare(metadata %struct.vec** {{.*}}, metadata ![[RANGE1:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[BEGIN1:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[END1:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata %struct.vec** {{.*}}, metadata ![[RANGE2:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[BEGIN2:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[END2:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata %struct.vec** {{.*}}, metadata ![[RANGE3:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[BEGIN3:[0-9]+]]
// CHECK: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[END3:[0-9]+]]
// CHECK: ![[RANGE1]] = !DILocalVariable(name: "__range1",
// CHECK: ![[BEGIN1]] = !DILocalVariable(name: "__begin1",
// CHECK: ![[END1]] = !DILocalVariable(name: "__end1",
// CHECK: ![[RANGE2]] = !DILocalVariable(name: "__range2",
// CHECK: ![[BEGIN2]] = !DILocalVariable(name: "__begin2",
// CHECK: ![[END2]] = !DILocalVariable(name: "__end2",
// CHECK: ![[RANGE3]] = !DILocalVariable(name: "__range3",
// CHECK: ![[BEGIN3]] = !DILocalVariable(name: "__begin3",
// CHECK: ![[END3]] = !DILocalVariable(name: "__end3",
