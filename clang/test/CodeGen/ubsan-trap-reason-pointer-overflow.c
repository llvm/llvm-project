// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s

#include <stddef.h>
#include <stdint.h>

int* pointer_overflow(void)
{
    int buf[4];
    volatile size_t n = (SIZE_MAX / sizeof(int)) - 1;
    return buf + n;
}

// CHECK: call void @llvm.ubsantrap(i8 19) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer