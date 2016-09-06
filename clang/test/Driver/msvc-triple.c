// RUN: %clang -target i686-pc-windows-msvc19 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=TARGET-19
// RUN: %clang -target i686-pc-windows-msvc   -S -emit-llvm %s -o - -fms-compatibility-version=19 | FileCheck %s --check-prefix=OVERRIDE-19
// RUN: %clang -target i686-pc-windows-msvc-elf -S -emit-llvm %s -o - | FileCheck %s --check-prefix=ELF-DEFAULT

// TARGET-19:   target triple = "i686-pc-windows-msvc19.0.0"
// OVERRIDE-19: target triple = "i686-pc-windows-msvc19.0.0"
// ELF-DEFAULT: target triple = "i686-pc-windows-msvc{{.*}}-elf"
