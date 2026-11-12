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

// PLAIN:     file format mach-o arm64
// PLAIN-NOT: __debug_

// RELOC:     RELOCATION RECORDS FOR [__debug_info]:
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_abbrev
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_line
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_str_offs
// RELOC-DAG: ARM64_RELOC_UNSIGNED __debug_addr

int g;
