// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=shift-exponent -fsanitize-trap=shift-exponent -emit-llvm %s -o - | FileCheck %s --check-prefix=RS

int right_shift_out_of_bounds(void) {
  int sh = 32;
  return 1 >> sh;
}

// RS-LABEL: @right_shift_out_of_bounds
// RS: call void @llvm.ubsantrap(i8 20) {{.*}}!dbg [[LOC:![0-9]+]]
// RS: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// RS: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Right shift is too large for 32-bit type 'int'"

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=shift-exponent -fsanitize-trap=shift-exponent -emit-llvm %s -o - | FileCheck %s --check-prefix=LS

int left_shift_out_of_bounds(void) {
  int sh = 32;
  return 1 << sh;
}

// LS-LABEL: @left_shift_out_of_bounds
// LS: call void @llvm.ubsantrap(i8 20) {{.*}}!dbg [[LOC:![0-9]+]]
// LS: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// LS: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Left shift is too large for 32-bit type 'int'"
