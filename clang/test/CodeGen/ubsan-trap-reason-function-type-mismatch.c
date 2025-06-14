// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=undefined \
// RUN: -fsanitize-trap=undefined -emit-llvm -S -c %s -o - | FileCheck %s

void target() { }

int function_type_mismatch() {
    int (*fp_int)(int);

    fp_int = (int (*)(int))(void *)target;

    return fp_int(42);
}

// CHECK: call void @llvm.ubsantrap(i8 6) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$