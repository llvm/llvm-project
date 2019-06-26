// RUN: %clang -Xclang -femit-debug-entry-values -g -O2 -S -target x86_64-none-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-ENTRY-VAL-OPT
// CHECK-ENTRY-VAL-OPT: !DILocalVariable(name: "a", arg: 1, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}})
// CHECK-ENTRY-VAL-OPT: !DILocalVariable(name: "b", arg: 2, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}}, flags: DIFlagArgumentNotModified)
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// CHECK-NOT: !DILocalVariable(name: "b", arg: 2, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}}, flags: DIFlagArgumentNotModified)
//

int fn2 (int a, int b) {
  ++a;
  return b;
}
