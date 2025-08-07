// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=builtin -fsanitize-trap=builtin -emit-llvm %s -o - | FileCheck %s

unsigned invalid_builtin(unsigned x) { return __builtin_clz(x); }

// CHECK-LABEL: @invalid_builtin
// CHECK: call void @llvm.ubsantrap(i8 8) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Invalid use of builtin function"
