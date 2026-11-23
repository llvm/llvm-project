// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=function -fsanitize-trap=function -emit-llvm %s -o - | FileCheck %s

void target(void) {}

int function_type_mismatch(void) {
  int (*fp_int)(int);

  fp_int = (int (*)(int))(void *)target;

  return fp_int(42);
}

// CHECK-LABEL: @target
// CHECK-LABEL: @function_type_mismatch
// CHECK: call void @llvm.ubsantrap(i8 6) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Function called with mismatched signature"
