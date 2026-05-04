// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=unreachable -fsanitize-trap=unreachable -emit-llvm %s -o - | FileCheck %s

int call_builtin_unreachable(void) { __builtin_unreachable(); }

// CHECK-LABEL: @call_builtin_unreachable
// CHECK: call void @llvm.ubsantrap(i8 1) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$_builtin_unreachable(), execution reached an unreachable program point"
