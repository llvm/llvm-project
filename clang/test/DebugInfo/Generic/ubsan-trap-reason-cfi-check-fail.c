// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm %s -o - | FileCheck %s

typedef int (*fp_t)(int);

int good(int x) { return x + 1; }

int bad(void) { return 0; }

int cfi_trigger(int a) {
  fp_t p = good;
  int r1 = p(a);

  p = (fp_t)(void *)bad;
  int r2 = p(a);

  return r1 + r2;
}

// CHECK-LABEL: @good
// CHECK-LABEL: @bad
// CHECK-LABEL: @cfi_trigger
// CHECK: call void @llvm.ubsantrap(i8 2) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Control flow integrity check failed"
