// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=nullability-arg -fsanitize-trap=nullability-arg -emit-llvm %s -o - | FileCheck %s

#include <stddef.h>

int nullability_arg(int *_Nonnull p) { return *p; }

int trigger_nullability_arg(void) { return nullability_arg(NULL); }

// CHECK-LABEL: @nullability_arg
// CHECK-LABEL: @trigger_nullability_arg
// CHECK: call void @llvm.ubsantrap(i8 14) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Passing null as an argument which is annotated with _Nonnull"
