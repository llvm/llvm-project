// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm %s -o - | FileCheck %s --check-prefix=SIO

int signed_mul_overflow(int a, int b) { return a * b; }

// SIO-LABEL: @signed_mul_overflow
// SIO: call void @llvm.ubsantrap(i8 12) {{.*}}!dbg [[LOC:![0-9]+]]
// SIO: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// SIO: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Signed integer multiplication overflow on type 'int'"

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=unsigned-integer-overflow -fsanitize-trap=unsigned-integer-overflow -emit-llvm %s -o - | FileCheck %s --check-prefix=UIO

unsigned unsigned_mul_overflow(unsigned a, unsigned b) { return a * b; }

// UIO-LABEL: @unsigned_mul_overflow
// UIO: call void @llvm.ubsantrap(i8 12) {{.*}}!dbg [[LOC:![0-9]+]]
// UIO: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// UIO: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Unsigned integer multiplication overflow on type 'unsigned int'"
