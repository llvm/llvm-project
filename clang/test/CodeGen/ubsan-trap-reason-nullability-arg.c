// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=nullability-arg \
// RUN: -fsanitize-trap=nullability-arg -emit-llvm -S -c %s -o - | FileCheck %s

#include <stddef.h>

int nullability_arg(int* _Nonnull p)
{
    return *p;
}

int trigger_nullability_arg()
{
    return nullability_arg(NULL);
}


// CHECK: call void @llvm.ubsantrap(i8 14) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$