// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=ANNOTATE

//==============================================================================
// Detailed trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons -emit-llvm %s -o - | FileCheck %s --check-prefixes=ANNOTATE,DETAILED

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons=detailed -emit-llvm %s -o - | FileCheck %s --check-prefixes=ANNOTATE,DETAILED

//==============================================================================
// Basic trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons=basic -emit-llvm %s -o - | FileCheck %s --check-prefixes=ANNOTATE,BASIC

//==============================================================================
// No trap reasons
//==============================================================================

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow \
// RUN: -fno-sanitize-debug-trap-reasons -emit-llvm %s -o - | FileCheck %s --check-prefix=NO-ANNOTATE

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons=none -emit-llvm %s -o - | FileCheck %s --check-prefix=NO-ANNOTATE

int add_overflow(int a, int b) { return a + b; }

// ANNOTATE-LABEL: @add_overflow
// ANNOTATE: call void @llvm.ubsantrap(i8 0) {{.*}}!dbg [[LOC:![0-9]+]]
// ANNOTATE: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// DETAILED: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$signed integer addition overflow in 'a + b'"
// BASIC: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Integer addition overflowed"

// NO-ANNOTATE-LABEL: @add_overflow
// NO-ANNOTATE: call void @llvm.ubsantrap(i8 0) {{.*}}!dbg [[LOC:![0-9]+]]
// NO-ANNOTATE-NOT: __clang_trap_msg
