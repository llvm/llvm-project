// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=return -fsanitize-trap=return -emit-llvm %s -o - | FileCheck %s

int missing_return(int x) {
  if (x > 0)
    return x;
}

// CHECK-LABEL: @_Z14missing_return
// CHECK: call void @llvm.ubsantrap(i8 11) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Execution reached the end of a value-returning function without returning a value"
