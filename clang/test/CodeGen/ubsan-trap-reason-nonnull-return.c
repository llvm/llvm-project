// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=returns-nonnull-attribute \
// RUN: -fsanitize-trap=returns-nonnull-attribute -emit-llvm -S -c %s -o - | FileCheck %s

__attribute__((returns_nonnull))
int* must_return_nonnull(int bad)
{
    if (bad)
        return 0;
    static int x = 1;
    return &x;
}

// CHECK: call void @llvm.ubsantrap(i8 17) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$UBSan Trap Reason