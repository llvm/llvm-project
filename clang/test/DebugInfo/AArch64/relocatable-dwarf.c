// -grelocatable-dwarf relocates offsets between DWARF sections in Mach-O
// object files.

// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-apple-macos15 -debug-info-kind=limited \
// RUN:   -dwarf-version=5 -emit-obj %s -o %t.o
// RUN: llvm-objdump -r --section=__debug_info %t.o \
// RUN:   | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple arm64-apple-macos15 -debug-info-kind=limited \
// RUN:   -dwarf-version=5 -grelocatable-dwarf -emit-obj %s -o %t.reloc.o
// RUN: llvm-objdump -r --section=__debug_info %t.reloc.o \
// RUN:   | FileCheck %s --check-prefix=RELOC

// The IR records the choice, so that code generation from IR honors it.
// RUN: %clang_cc1 -triple arm64-apple-macos15 -debug-info-kind=limited \
// RUN:   -dwarf-version=5 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=PLAIN-IR
// RUN: %clang_cc1 -triple arm64-apple-macos15 -debug-info-kind=limited \
// RUN:   -dwarf-version=5 -grelocatable-dwarf -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=RELOC-IR < %t.ll
// RUN: %clang_cc1 -triple arm64-apple-macos15 -emit-obj -x ir %t.ll \
// RUN:   -o %t.ir.o
// RUN: llvm-objdump -r --section=__debug_info %t.ir.o \
// RUN:   | FileCheck %s --check-prefix=RELOC

// PLAIN:     file format mach-o arm64
// PLAIN-NOT: __debug_

// RELOC:     RELOCATION RECORDS FOR [__debug_info]:
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_abbrev
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_line
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_str_offs
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_addr

// PLAIN-IR:     !llvm.module.flags =
// PLAIN-IR-NOT: MachO Relocatable DWARF

// RELOC-IR: !{i32 7, !"MachO Relocatable DWARF", i32 1}

int g;
