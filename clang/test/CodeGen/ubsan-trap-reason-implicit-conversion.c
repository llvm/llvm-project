// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=implicit-conversion \
// RUN: -fsanitize-trap=implicit-conversion -emit-llvm -S -c %s -o - | FileCheck %s

unsigned long long big; 

unsigned implicit_conversion()
{
    return big;
}

// CHECK: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$