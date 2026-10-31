// REQUIRES: x86-registered-target

// RUN: rm -rf %t; mkdir %t

// RUN: %clang -target x86_64-apple-macos11.0 -c %s -o %t/x86_64.o -femit-compact-unwind-non-canonical
// RUN: llvm-objdump --macho --dwarf=frames %t/x86_64.o | FileCheck %s --check-prefix=WITH-FDE

// RUN: %clang -target x86_64-apple-macos11.0 -femit-dwarf-unwind=no-compact-unwind -c %s -o %t/x86_64-no-dwarf.o -femit-compact-unwind-non-canonical
// RUN: llvm-objdump --macho --dwarf=frames %t/x86_64-no-dwarf.o | FileCheck %s --check-prefix=NO-FDE

// RUN: %clang -target x86_64-apple-macos11.0 -femit-dwarf-unwind=dwarf-only -c %s -o %t/x86_64-dwarf-only.o
// RUN: llvm-objdump --macho --dwarf=frames --unwind-info %t/x86_64-dwarf-only.o | FileCheck %s --check-prefixes=WITH-FDE,CU-DWARF

// CU-DWARF:      Contents of __compact_unwind section:
// CU-DWARF-NEXT:   Entry at offset 0x0:
// CU-DWARF-NEXT:     start:                0x0 _foo
// CU-DWARF-NEXT:     length:               0x1
// CU-DWARF-NEXT:     compact encoding:     0x04000000

// WITH-FDE: FDE
// NO-FDE-NOT: FDE

.text
_foo:
  .cfi_startproc
  .cfi_def_cfa_offset 8
  ret
  .cfi_endproc
