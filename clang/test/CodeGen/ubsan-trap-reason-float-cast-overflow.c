// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -debug-info-kind=standalone -dwarf-version=5 -fsanitize=float-cast-overflow \
// RUN: -fsanitize-trap=float-cast-overflow -emit-llvm %s -o - | FileCheck %s

int f(float x) { 
  return (int)x; 
}

// CHECK: call void @llvm.ubsantrap(i8 5) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$