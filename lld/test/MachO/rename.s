# REQUIRES: x86
# RUN: rm -fr %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o

## Check option format
# RUN: not %lld \
# RUN:     -rename_section B@GUS_SEG b@gus_sect S/ASHY_SEG st*rry_sect \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT \
# RUN:   -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=BAD1

# BAD1-DAG: error: invalid name for segment or section: B@GUS_SEG
# BAD1-DAG: error: invalid name for segment or section: b@gus_sect
# BAD1-DAG: error: invalid name for segment or section: S/ASHY_SEG
# BAD1-DAG: error: invalid name for segment or section: st*rry_sect
# BAD1-DAG: error: invalid name for segment or section: -o
# BAD1-DAG: error: /dev/null: unhandled file type

# RUN: not %lld \
# RUN:     -rename_segment H#SHY_SEG PL+SSY_SEG \
# RUN:     -rename_segment __FROM_SEG \
# RUN:   -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=BAD2

# BAD2-DAG: error: invalid name for segment or section: H#SHY_SEG
# BAD2-DAG: error: invalid name for segment or section: PL+SSY_SEG
# BAD2-DAG: error: invalid name for segment or section: -o
# BAD2-DAG: error: /dev/null: unhandled file type

## Check that section and segment renames happen
# RUN: %lld \
# RUN:     -rename_section __FROM_SECT __from_sect __TO_SECT __to_sect \
# RUN:     -rename_segment __FROM_SEG __TO_SEG \
# RUN:   -o %t %t.o
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s

# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __text
# CHECK-NEXT: segname __TEXT
# CHECK:      {{^}}Section{{$}}
# CHECK-NOT:  sectname __from_sect
# CHECK-NEXT: sectname __to_sect
# CHECK-NOT:  segname __FROM_SECT
# CHECK-NEXT: segname __TO_SECT
# CHECK:      {{^}}Section{{$}}
# CHECK-NEXT: sectname __from_seg
# CHECK-NOT:  segname __FROM_SEG
# CHECK-NEXT: segname __TO_SEG

.section __FROM_SECT,__from_sect
.global _from_sect
_from_sect:
  .space 8

.section __FROM_SEG,__from_seg
.global _from_seg
_from_seg:
  .space 8

.text
.global _main
_main:
  ret
