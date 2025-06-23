// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=builtin \
// RUN: -fsanitize-trap=builtin -emit-llvm -S -c %s -o - | FileCheck %s

unsigned invalid_builtin(unsigned x)
{
    return __builtin_clz(x);
}


// CHECK: call void @llvm.ubsantrap(i8 8) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$UBSan Trap Reason