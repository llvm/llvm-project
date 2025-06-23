// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=nullability-return \
// RUN: -fsanitize-trap=nullability-return -emit-llvm -S -c %s -o - | FileCheck %s

#include <stdbool.h>
#include <stddef.h>

int* _Nonnull nullability_return(bool fail)
{
    if (fail)
        return NULL;

    static int x = 0;
    return &x;
}


// CHECK: call void @llvm.ubsantrap(i8 15) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$UBSan Trap Reason