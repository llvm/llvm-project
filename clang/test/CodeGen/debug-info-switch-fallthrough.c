// RUN:  %clang_cc1 -triple x86_64-apple-macosx11.0.0 -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s
// CHECK: ], !dbg !{{[0-9]+}}
// CHECK-EMPTY:
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: br {{.+}}, !dbg !{{[0-9+]}}
// CHECK-EMPTY:
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: br {{.+}}, !dbg ![[LOC:[0-9]+]]
void test(int num) {
  switch (num) {
  case 0:
    break;
  case 10: // CHECK: ![[LOC]] = !DILocation(line: [[@LINE]], column:{{.+}}, scope: {{.+}})
  default:
    break;
  }
}
