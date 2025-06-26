// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=bool -fsanitize-trap=bool -emit-llvm %s -o - | FileCheck %s

#include <stdbool.h> 

unsigned char bad_byte;

bool load_invalid_value()
{
    return *((bool *)&bad_byte);
}

// CHECK: call void @llvm.ubsantrap(i8 10) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer