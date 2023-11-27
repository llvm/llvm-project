# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-android %s -o %t
# RUN: ld.lld %t -o %t.so -shared --android-memtag-mode=sync

## Normally relocations are printed before the symbol tables, so reorder it a
## bit to make it easier on matching addresses of relocations up with the
## symbols.
# RUN: llvm-readelf %t.so -s > %t.out
# RUN: llvm-readelf %t.so --section-headers --relocs --memtag >> %t.out
# RUN: FileCheck %s < %t.out

# CHECK:     Symbol table '.dynsym' contains
# CHECK-DAG: [[#%x,GLOBAL:]] 16 OBJECT GLOBAL DEFAULT [[#]] global{{$}}
# CHECK-DAG: [[#%x,GLOBAL_IN_SECTION:]] 16 OBJECT GLOBAL DEFAULT [[#]] global_in_section{{$}}

# CHECK:     Section Headers:
# CHECK:     .memtag.globals.dynamic AARCH64_MEMTAG_GLOBALS_DYNAMIC
# CHECK-NOT: .memtag.globals.static
# CHECK-NOT: AARCH64_MEMTAG_GLOBALS_STATIC

# CHECK:      Memtag Dynamic Entries
# CHECK-NEXT: AARCH64_MEMTAG_MODE: Synchronous (0)
# CHECK-NEXT: AARCH64_MEMTAG_HEAP: Disabled (0)
# CHECK-NEXT: AARCH64_MEMTAG_STACK: Disabled (0)
# CHECK-NEXT: AARCH64_MEMTAG_GLOBALS: 0x{{[0-9a-f]+}}
# CHECK-NEXT: AARCH64_MEMTAG_GLOBALSSZ: 3

# CHECK:      Memtag Global Descriptors:
# CHECK-NEXT: 0x[[#GLOBAL]]: 0x10
# CHECK-NOT:  0x

        .memtag   global
        .type     global,@object
        .bss
        .globl    global
        .p2align  4, 0x0
global:
        .zero     16
        .size     global, 16

        .memtag   global_in_section                  // @global_in_section
        .type     global_in_section,@object
        .section  metadata_strings,"aw",@progbits
        .globl    global_in_section
        .p2align  4, 0x0
global_in_section:
        .zero     16
        .size     global_in_section, 16
