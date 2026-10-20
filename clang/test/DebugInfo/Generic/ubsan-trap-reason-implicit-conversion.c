// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=implicit-unsigned-integer-truncation -fsanitize-trap=implicit-unsigned-integer-truncation -emit-llvm %s -o - | FileCheck %s

unsigned long long big;

unsigned implicit_conversion(void) { return big; }

// CHECK-LABEL: @implicit_conversion
// CHECK: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Implicit integer conversion overflowed or lost data"
