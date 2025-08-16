// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm %s -o - | FileCheck %s --check-prefix=SIO

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=integer-divide-by-zero -fsanitize-trap=integer-divide-by-zero -emit-llvm %s -o - | FileCheck %s --check-prefix=DBZ

int div_rem_overflow(int a, int b) { return a / b; }

// SIO-LABEL: @div_rem_overflow
// SIO: call void @llvm.ubsantrap(i8 3) {{.*}}!dbg [[LOC:![0-9]+]]
// SIO: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// SIO: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Signed integer division overflow on type 'int'"

// DBZ-LABEL: @div_rem_overflow
// DBZ: call void @llvm.ubsantrap(i8 3) {{.*}}!dbg [[LOC:![0-9]+]]
// DBZ: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// DBZ: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Division by zero on type 'int'"
