// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow \
// RUN: -fsanitize-trap=signed-integer-overflow,unsigned-integer-overflow \
// RUN: -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,DETAILED %s

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow \
// RUN: -fsanitize-trap=signed-integer-overflow,unsigned-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons=basic \
// RUN: -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,BASIC %s

int ssub_overflow(int a, int b) { return a - b; }

unsigned sub_overflow(unsigned c, unsigned d) { return c - d; }

// CHECK-LABEL: @ssub_overflow
// CHECK: call void @llvm.ubsantrap(i8 21) {{.*}}!dbg [[SLOC:![0-9]+]]

// CHECK-LABEL: @sub_overflow
// CHECK: call void @llvm.ubsantrap(i8 21) {{.*}}!dbg [[LOC:![0-9]+]]

// DETAILED: [[SLOC]] = !DILocation(line: 0, scope: [[SMSG:![0-9]+]], {{.+}})
// DETAILED: [[SMSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$signed integer subtraction overflow in 'a - b'"
// DETAILED: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// DETAILED: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$unsigned integer subtraction overflow in 'c - d'"

// In "Basic" mode the trap reason is shared by both functions.
// BASIC: [[SLOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// BASIC: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Integer subtraction overflowed"
// BASIC: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
