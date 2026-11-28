// Check that the AST file is read correctly from a COFF PCM, even when the
// clangast section contents aren't aligned within the file.

// REQUIRES: aarch64-registered-target
// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsanitize=address \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fmodule-format=obj -fno-modules-global-index -fsyntax-only \
// RUN:   %t/t.c -I %t

// The MC COFF writer aligns section contents within the file, but llvm-objcopy
// lays them out without any padding. Use it to misalign the clangast section.
// RUN: llvm-objcopy %t/cache/*/M-*.pcm
// RUN: llvm-readobj --sections %t/cache/*/M-*.pcm | FileCheck %s
// CHECK:      Name: clangast
// CHECK-NEXT: VirtualSize:
// CHECK-NEXT: VirtualAddress:
// CHECK-NEXT: RawDataSize:
// CHECK-NEXT: PointerToRawData: 0x{{[0-9A-F]*}}{{[1235679ABDEF]}}{{$}}

// Read the PCM from disk.
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsanitize=address \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fmodule-format=obj -fno-modules-global-index -fsyntax-only \
// RUN:   %t/t.c -I %t -Rmodule-import 2>&1 | FileCheck %s --check-prefix=IMPORT
// IMPORT: remark: importing module 'M' from '{{.*}}M-{{.*}}.pcm'
// IMPORT-NOT: remark: building module

// Build the global module index, which reads the PCM from disk too.
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsanitize=address \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -fmodule-format=obj -fsyntax-only %t/t.c -I %t

//--- module.modulemap
module M { header "m.h" }
//--- m.h
int m(void);
//--- t.c
#include "m.h"
