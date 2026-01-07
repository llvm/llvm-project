// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=returns-nonnull-attribute -fsanitize-trap=returns-nonnull-attribute -emit-llvm %s -o - | FileCheck %s

__attribute__((returns_nonnull)) int *must_return_nonnull(int bad) {
  if (bad)
    return 0;
  static int x = 1;
  return &x;
}

// CHECK-LABEL: @must_return_nonnull
// CHECK: call void @llvm.ubsantrap(i8 17) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Returning null pointer from a function which is declared to never return null"
