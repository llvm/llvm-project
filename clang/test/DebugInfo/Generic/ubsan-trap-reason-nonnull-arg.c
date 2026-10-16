// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=nonnull-attribute -fsanitize-trap=nonnull-attribute -emit-llvm %s -o - | FileCheck %s

__attribute__((nonnull)) void nonnull_arg(int *p) { (void)p; }

void trigger_nonnull_arg() { nonnull_arg(0); }

// CHECK-LABEL: @nonnull_arg
// CHECK-LABEL: @trigger_nonnull_arg
// CHECK: call void @llvm.ubsantrap(i8 16) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Passing null pointer as an argument which is declared to never be null"
