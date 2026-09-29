// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=vla-bound -fsanitize-trap=vla-bound -emit-llvm %s -o - | FileCheck %s

int n = 0;

int vla_bound_not_positive(void) {
  int a[n];
  return sizeof a;
}

// CHECK-LABEL: @vla_bound_not_positive
// CHECK: call void @llvm.ubsantrap(i8 24) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Variable length array bound evaluates to non-positive value"
