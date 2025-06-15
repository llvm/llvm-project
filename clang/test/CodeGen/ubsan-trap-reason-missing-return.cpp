// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=return \
// RUN: -fsanitize-trap=return -emit-llvm -S -c %s -o - | FileCheck %s

int missing_return(int x)
{
    if (x > 0)
        return x;
}

// CHECK: call void @llvm.ubsantrap(i8 11) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$