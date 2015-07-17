@import DependsOnModule;
// REQUIRES: x86-registered-target
// RUN: rm -rf %t-MachO %t-ELF %t-ELF_SPLIT %t-COFF %t-raw
// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fmodules -fmodule-format=obj -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-MachO -F %S/Inputs %s
// RUN: %clang_cc1 -triple=x86_64-linux-elf -fmodules -fmodule-format=obj -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-ELF -F %S/Inputs %s
// RUN: %clang_cc1 -triple=x86_64-windows-coff -fmodules -fmodule-format=obj -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-COFF -F %S/Inputs %s
// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fmodules -fmodule-format=raw -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-raw -F %S/Inputs %s


// RUN: llvm-objdump -section-headers %t-MachO/DependsOnModule.pcm %t-ELF/DependsOnModule.pcm %t-COFF/DependsOnModule.pcm | FileCheck %s
// CHECK: file format Mach-O 64-bit x86-64
// CHECK: __clangast   {{[0-9a-f]+}} {{[0-9a-f]+}} DATA
// CHECK: file format ELF64-x86-64
// CHECK: __clangast   {{[0-9a-f]+}} {{[0-9a-f]+}} DATA
// CHECK: file format COFF-x86-64
// CHECK: clangast   {{[0-9a-f]+}} {{[0-9a-f]+}}

// RUN: not llvm-objdump -section-headers %t-raw/DependsOnModule.pcm

// RUN: %clang_cc1 -split-dwarf-file t-split.dwo -triple=x86_64-linux-elf -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-ELF_SPLIT -F %S/Inputs %s -o %t-split.o
