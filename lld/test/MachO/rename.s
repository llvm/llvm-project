# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t.o
# RUN: %lld -o %t.out %t.o

## Check option format.
# RUN: not %lld \
# RUN:     -rename_section B@GUS_SEG b@gus_sect S/ASHY_SEG st*rry_sect \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT \
# RUN:   -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=BAD1

# BAD1-DAG: error: invalid name for segment or section: B@GUS_SEG
# BAD1-DAG: error: invalid name for segment or section: b@gus_sect
# BAD1-DAG: error: invalid name for segment or section: S/ASHY_SEG
# BAD1-DAG: error: invalid name for segment or section: st*rry_sect
# BAD1-DAG: error: invalid name for segment or section: -o
# BAD1-DAG: error: {{.*}}: unhandled file type

# RUN: not %lld \
# RUN:     -rename_segment H#SHY_SEG PL+SSY_SEG \
# RUN:     -rename_segment __FROM_SEG \
# RUN:   -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=BAD2

# BAD2-DAG: error: invalid name for segment or section: H#SHY_SEG
# BAD2-DAG: error: invalid name for segment or section: PL+SSY_SEG
# BAD2-DAG: error: invalid name for segment or section: -o
# BAD2-DAG: error: {{.*}}: unhandled file type

## Check that section and segment renames happen.
# RUN: %lld -lSystem \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT __to_sect \
# RUN:     -rename_segment __FROM_SEG __TO_SEG \
# RUN:     -rename_section __TEXT __cstring __RODATA __cstring \
# RUN:   -o %t.out %t.o
# RN: llvm-otool -l %t.out | FileCheck %s

# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __text
# CHECK-NEXT: segname __TEXT
# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __to_sect
# CHECK-NEXT: segname __TO_SECT
# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __from_seg
# CHECK-NEXT: segname __TO_SEG
# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __cstring
# CHECK-NEXT: segname __RODATA

## Check interaction between -rename_section and -rename_segment.
## rename_segment should be applied after rename_section, so the output
## name of rename_section is renamed by rename_segment.
## (ld64 leaves an empty __TO_SECT,__to_sect in the output for the intermediate
## name, but it too writes the actual data to __SEG,__to_sect.)
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/small.s \
# RUN:     -o %t/small.o
# RUN: %lld -dylib \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT __to_sect \
# RUN:     -rename_segment __TO_SECT __SEG \
# RUN:   -o %t.dylib %t/small.o
# RUN: llvm-otool -l %t.dylib | FileCheck --check-prefix=SECTSEGYES %s
# RUN: %lld -dylib \
# RUN:     -rename_segment __TO_SECT __SEG \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT __to_sect \
# RUN:   -o %t.dylib %t/small.o
# RUN: llvm-otool -l %t.dylib | FileCheck --check-prefix=SECTSEGYES %s
# SECTSEGYES:      Section
# SECTSEGYES-NEXT:   sectname __text
# SECTSEGYES-NEXT:    segname __TEXT
# SECTSEGYES:      Section
# SECTSEGYES-NEXT:   sectname __to_sect
# SECTSEGYES-NEXT:    segname __SEG
## ...but rename_segment has no effect if it doesn't match the name after
## rename_section is applied.
# RUN: %lld -dylib \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT __to_sect \
# RUN:     -rename_segment __FROM_SECT __SEG \
# RUN:   -o %t.dylib %t/small.o
# RUN: llvm-otool -l %t.dylib | FileCheck --check-prefix=SECTSEGSOME %s
# SECTSEGSOME:      Section
# SECTSEGSOME-NEXT:   sectname __text
# SECTSEGSOME-NEXT:    segname __TEXT
# SECTSEGSOME:      Section
# SECTSEGSOME-NEXT:   sectname __to_sect
# SECTSEGSOME-NEXT:    segname __TO_SECT
## If rename_section would only match after rename_segment, rename_section has
## no effect.
# RUN: %lld -dylib \
# RUN:     -rename_section __SEG __from_sect __TO_SECT __to_sect \
# RUN:     -rename_segment __FROM_SECT __SEG \
# RUN:   -o %t.dylib %t/small.o
# RUN: llvm-otool -l %t.dylib | FileCheck --check-prefix=SECTSEGNO %s
# RUN: %lld -dylib \
# RUN:     -rename_segment __FROM_SECT __SEG \
# RUN:     -rename_section __SEG __from_sect __TO_SECT __to_sect \
# RUN:   -o %t.dylib %t/small.o
# RUN: llvm-otool -l %t.dylib | FileCheck --check-prefix=SECTSEGNO %s
# SECTSEGNO:      Section
# SECTSEGNO-NEXT:   sectname __text
# SECTSEGNO-NEXT:    segname __TEXT
# SECTSEGNO:      Section
# SECTSEGNO-NEXT:   sectname __from_sect
# SECTSEGNO-NEXT:    segname __SEG

#--- main.s
.section __FROM_SECT,__from_sect
.global _from_sect
_from_sect:
  .space 8

.section __FROM_SEG,__from_seg
.global _from_seg
_from_seg:
  .space 8

## This is a synthetic section; make sure it gets renamed too.
.cstring
  .space 8

.text
.global _main
_main:
  ret

#--- small.s
.section __FROM_SECT,__from_sect
.global _from_sect
_from_sect:
  .space 8
