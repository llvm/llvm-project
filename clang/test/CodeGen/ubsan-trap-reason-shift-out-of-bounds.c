// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=shift \
// RUN: -fsanitize-trap=shift -emit-llvm -S -c %s -o - | FileCheck %s

int shift_out_of_bounds()
{
    int sh = 32;
    return 1 << sh;
}

// CHECK: call void @llvm.ubsantrap(i8 20) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$