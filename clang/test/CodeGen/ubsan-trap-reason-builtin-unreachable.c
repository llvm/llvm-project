// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=unreachable \
// RUN: -fsanitize-trap=unreachable -emit-llvm -S -c %s -o - | FileCheck %s

int call_builtin_unreachable()
{
    __builtin_unreachable();
}


// CHECK: call void @llvm.ubsantrap(i8 1) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$UBSan Trap Reason