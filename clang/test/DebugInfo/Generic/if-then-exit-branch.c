// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s

// The unconditional branch from the 'then' (and 'else') block to the
// continuation block is not a statement, so it must not carry a debug
// location.

int g;

// CHECK-LABEL: define {{.*}}@no_else(
// CHECK:       if.then:
// CHECK:         store i32 1, ptr @g{{.*}}, !dbg
// CHECK-NEXT:    br label %if.end{{$}}
void no_else(int a) {
  if (a) {
    g = 1;
  }
  g = 2;
}

// CHECK-LABEL: define {{.*}}@with_else(
// CHECK:       if.then:
// CHECK:         store i32 1, ptr @g{{.*}}, !dbg
// CHECK-NEXT:    br label %if.end{{$}}
// CHECK:       if.else:
// CHECK:         store i32 2, ptr @g{{.*}}, !dbg
// CHECK-NEXT:    br label %if.end{{$}}
void with_else(int a) {
  if (a) {
    g = 1;
  } else {
    g = 2;
  }
  g = 3;
}
