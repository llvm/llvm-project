// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -S -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-DEF,EXPLICIT-DEF %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=none -S -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-DEF,EXPLICIT-DEF %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=explicit -S -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-DEF,EXPLICIT-EXP %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=all -S -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-EXP,EXPLICIT-EXP %s
// RUN: %clang -target powerpc-ibm-aix %s -mdefault-visibility-export-mapping=all -fvisibility=hidden -S -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-HID,EXPLICIT-EXP %s

// RUN: not %clang -mdefault-visibility-export-mapping=explicit -target powerpc-unknown-linux %s 2>&1 | \
// RUN: FileCheck -check-prefix=ERROR %s
// ERROR: unsupported option '-mdefault-visibility-export-mapping=explicit' for target 'powerpc-unknown-linux'

// UNSPECIFIED-DEF: define void @func()
// UNSPECIFIED-HID: define hidden void @func()
// UNSPECIFIED-EXP: define dllexport void @func()
void func() {}

#pragma GCC visibility push(default)
// EXPLICIT-DEF: define void @pragmafunc()
// EXPLICIT-EXP: define dllexport void @pragmafunc()
void pragmafunc() {}
#pragma GCC visibility pop

// EXPLICIT-DEF: define void @explicitfunc()
// EXPLICIT-EXP: define dllexport void @explicitfunc()
void __attribute__((visibility("default"))) explicitfunc() {}
