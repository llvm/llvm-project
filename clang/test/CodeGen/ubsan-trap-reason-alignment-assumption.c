// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - | FileCheck %s

#include <stdint.h>
int32_t *get_int(void) __attribute__((assume_aligned(16)));

void retrieve_int(void) {
  int *i = get_int();
  *i = 7;
}

// CHECK-LABEL: @retrieve_int
// CHECK: call void @llvm.ubsantrap(i8 23) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Alignment assumption violated"
