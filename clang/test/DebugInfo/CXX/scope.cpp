// RUN: %clang_cc1 -debug-info-kind=limited -gno-column-info -std=c++11 -emit-llvm %s -o -| FileCheck %s
//
// Two variables with the same name in subsequent if staments need to be in separate scopes.

int src();

void f();

void func() {
  // CHECK: = !DILocalVariable(name: "i"
  // CHECK-SAME:               scope: [[IF1:![0-9]*]]
  // CHECK-SAME:               line: [[@LINE+2]]
  // CHECK: [[IF1]] = distinct !DILexicalBlock({{.*}}line: [[@LINE+1]])
  if (int i = src())
    f();

  // CHECK: = !DILocalVariable(name: "i"
  // CHECK-SAME:               scope: [[IF2:![0-9]*]]
  // CHECK-SAME:               line: [[@LINE+2]]
  // CHECK: [[IF2]] = distinct !DILexicalBlock({{.*}}line: [[@LINE+1]])
  if (int i = src()) {
    f();
  } else
    f();

  // CHECK: = !DILocalVariable(name: "i"
  // CHECK-SAME:               scope: [[FOR:![0-9]*]]
  // CHECK-SAME:               line: [[@LINE+2]]
  // CHECK: [[FOR]] = distinct !DILexicalBlock({{.*}}line: [[@LINE+1]])
  for (int i = 0;
  // CHECK: = !DILocalVariable(name: "b"
  // CHECK-SAME:               scope: [[FOR_BODY:![0-9]*]]
  // CHECK-SAME:               line: [[@LINE+6]]
  // CHECK: [[FOR_BODY]] = distinct !DILexicalBlock({{.*}}line: [[@LINE-4]])
  // The scope could be located at 'bool b', but LLVM drops line information for
  // scopes anyway, so it's not terribly important.
  // FIXME: change the debug info schema to not include locations of scopes,
  // since they're not used.
       bool b = i != 10; ++i)
    f();

  // CHECK: = !DILocalVariable(name: "i"
  // CHECK-SAME:               scope: [[FOR:![0-9]*]]
  // CHECK-SAME:               line: [[@LINE+2]]
  // CHECK: [[FOR]] = distinct !DILexicalBlock({{.*}}line: [[@LINE+1]])
  for (int i = 0; i != 10; ++i) {
    // FIXME: Do not include scopes that have only other scopes (and no variables
    // or using declarations) as direct children, they just waste
    // space/relocations/etc.
    // CHECK: [[FOR_LOOP_INCLUDING_COND:!.*]] = distinct !DILexicalBlock(scope: [[FOR]],{{.*}} line: [[@LINE-4]])
    // CHECK: = !DILocalVariable(name: "b"
    // CHECK-SAME:               scope: [[FOR_COMPOUND:![0-9]*]]
    // CHECK-SAME:               line: [[@LINE+2]]
    // CHECK: [[FOR_COMPOUND]] = distinct !DILexicalBlock(scope: [[FOR_LOOP_INCLUDING_COND]],{{.*}} line: [[@LINE-8]])
    bool b = i % 2;
  }

  int x[] = {1, 2};
  // CHECK: = !DILocalVariable(name: "__range1"
  // CHECK-SAME:               scope: [[RANGE_FOR:![0-9]*]]
  // CHECK-NOT:                line:
  // CHECK-SAME:               ){{$}}
  // CHECK: [[RANGE_FOR]] = distinct !DILexicalBlock({{.*}}, line: [[@LINE+1]])
  for (int i : x) {
    // CHECK: = !DILocalVariable(name: "i"
    // CHECK-SAME:               scope: [[RANGE_FOR_BODY:![0-9]*]]
    // CHECK-SAME:               line: [[@LINE-3]]
    // CHECK: [[RANGE_FOR_BODY]] = distinct !DILexicalBlock(scope: [[RANGE_FOR]],{{.*}} line: [[@LINE-4]])
  }
}
