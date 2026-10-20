// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=float-cast-overflow -fsanitize-trap=float-cast-overflow -emit-llvm %s -o - | FileCheck %s

int float_cast_overflow(float x) { return (int)x; }

// CHECK-LABEL: @float_cast_overflow
// CHECK: call void @llvm.ubsantrap(i8 5) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Floating-point to integer conversion overflowed"
