// RUN: %clang_cc1 -triple x86_64-unknown_unknown -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
void ext();

struct Bar {
  void bar() { ext(); }
};

struct Foo {
  Bar *b;

  Bar *foo() { return b; }
};

void test(Foo *f) {
  f->foo()->bar();
}

// CHECK-LABEL: @_Z4testP3Foo
// CHECK: call {{.*}} @_ZN3Foo3fooEv{{.*}}, !dbg ![[CALL1LOC:.*]]
// CHECK: call void @_ZN3Bar3barEv{{.*}}, !dbg ![[CALL2LOC:.*]]

// CHECK: ![[CALL1LOC]] = !DILocation(line: [[LINE:[0-9]+]], column: 6,
// CHECK: ![[CALL2LOC]] = !DILocation(line: [[LINE]], column: 13,

