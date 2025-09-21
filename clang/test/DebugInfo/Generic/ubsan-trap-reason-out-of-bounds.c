// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm %s -o - | FileCheck %s

int out_of_bounds() {
  int a[1] = {0};
  return a[1];
}

// CHECK-LABEL: @out_of_bounds
// CHECK: call void @llvm.ubsantrap(i8 18) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Array index out of bounds"
