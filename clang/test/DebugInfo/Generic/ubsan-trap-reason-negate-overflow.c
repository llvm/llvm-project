// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm %s -o - | FileCheck %s

int negate_overflow() {
  int x;
  return -x;
}

// CHECK-LABEL: @negate_overflow
// CHECK: call void @llvm.ubsantrap(i8 13) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Integer negation overflowed"
