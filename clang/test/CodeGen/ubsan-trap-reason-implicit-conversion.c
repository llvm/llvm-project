// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 -fsanitize=implicit-signed-integer-truncation \
// RUN: -fsanitize-trap=implicit-signed-integer-truncation -emit-llvm %s -o - | FileCheck %s --check-prefix=SIC

long long signedBig;

int signed_implicit_conversion_truncation(void) { return signedBig; }

// SIC-LABEL: @signed_implicit_conversion_truncation
// SIC: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC:![0-9]+]]
// SIC: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// SIC: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Implicit signed conversion from 'long long' to 'int' caused truncation"

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 -fsanitize=implicit-unsigned-integer-truncation \
// RUN: -fsanitize-trap=implicit-unsigned-integer-truncation -emit-llvm %s -o - | FileCheck %s --check-prefix=UIC

unsigned long long unsignedBig;

unsigned unsigned_implicit_conversion_truncation(void) { return unsignedBig; }

// UIC-LABEL: @unsigned_implicit_conversion_truncation
// UIC: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC:![0-9]+]]
// UIC: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// UIC: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Implicit unsigned conversion from 'unsigned long long' to 'unsigned int' caused truncation"

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 -fsanitize=implicit-integer-sign-change \
// RUN: -fsanitize-trap=implicit-integer-sign-change -emit-llvm %s -o - | FileCheck %s --check-prefix=ISCUFS

unsigned to_unsigned_from_signed(int x) { return x; }

// ISCUFS-LABEL: @to_unsigned_from_signed
// ISCUFS: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC1:![0-9]+]]
// ISCUFS: [[LOC1]] = !DILocation(line: 0, scope: [[MSG1:![0-9]+]], {{.+}})
// ISCUFS: [[MSG1]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Implicit conversion from 'int' to 'unsigned int' caused sign-change"

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 -fsanitize=implicit-integer-sign-change \
// RUN: -fsanitize-trap=implicit-integer-sign-change -emit-llvm %s -o - | FileCheck %s --check-prefix=ISCSFU

int to_signed_from_unsigned(unsigned x) { return x; }

// ISCSFU-LABEL: @to_signed_from_unsigned
// ISCSFU: call void @llvm.ubsantrap(i8 7) {{.*}}!dbg [[LOC2:![0-9]+]]
// ISCSFU: [[LOC2]] = !DILocation(line: 0, scope: [[MSG2:![0-9]+]], {{.+}})
// ISCSFU: [[MSG2]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Implicit conversion from 'unsigned int' to 'int' caused sign-change"
