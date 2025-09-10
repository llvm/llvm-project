// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - | FileCheck %s

int type_mismatch(int *p) { return *p; }

// CHECK-LABEL: @type_mismatch
// CHECK: call void @llvm.ubsantrap(i8 22) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Type mismatch in operation"
