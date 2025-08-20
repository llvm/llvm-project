// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=nullability-return -fsanitize-trap=nullability-return -emit-llvm %s -o - | FileCheck %s

#include <stdbool.h>
#include <stddef.h>

int *_Nonnull nullability_return(bool fail) {
  if (fail)
    return NULL;

  static int x = 0;
  return &x;
}

// CHECK-LABEL: @nullability_return
// CHECK: call void @llvm.ubsantrap(i8 15) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Returning null from a function with a return type annotated with _Nonnull"
