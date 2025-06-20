// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=undefined \
// RUN: -fsanitize-trap=undefined -emit-llvm -S -c %s -o - | FileCheck %s

int add_overflow(int a, int b) {
  return a + b;
}

// CHECK: call void @llvm.ubsantrap(i8 0) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$
