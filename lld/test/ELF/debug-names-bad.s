# REQUIRES: x86, system-linux
# RUN: rm -rf %t && mkdir %t && cd %t

## Test errors in the header.
# RUN: sed '/Header: version/s/5/4/' %S/Inputs/debug-names-a.s > bad-version.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-version.s -o bad-version.o
# RUN: not ld.lld --debug-names bad-version.o 2>&1 | FileCheck %s --check-prefix=BAD-VERSION --implicit-check-not=error:

# BAD-VERSION: error: bad-version.o:(.debug_names): unsupported version: 4

# RUN: sed '/Header: name count/s/[0-9]/4/' %S/Inputs/debug-names-a.s > bad-name-count.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-name-count.s -o bad-name-count.o
# RUN: not ld.lld --debug-names bad-name-count.o 2>&1 | FileCheck %s --check-prefix=BAD-NAME-COUNT --implicit-check-not=error:

## Test errors in offsets.
# BAD-NAME-COUNT: error: bad-name-count.o:(.debug_names): Section too small: cannot read abbreviations.

# RUN: sed '/Offset in Bucket/s/long/byte/' %S/Inputs/debug-names-a.s > entry-offset-in-byte.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 entry-offset-in-byte.s -o entry-offset-in-byte.o
# RUN: not ld.lld --debug-names entry-offset-in-byte.o 2>&1 | FileCheck %s --check-prefix=ENTRY-OFFSET-IN-BYTE --implicit-check-not=error:

# ENTRY-OFFSET-IN-BYTE-COUNT-2: error: entry-offset-in-byte.o:(.debug_names): index entry is out of bounds

## Test errors in the abbrev table.
# RUN: sed -E '/DW_IDX_parent/{n;s/[0-9]+.*DW_FORM_flag_present/16/}' %S/Inputs/debug-names-a.s > bad-parent-form.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-parent-form.s -o bad-parent-form.o
# RUN: not ld.lld --debug-names bad-parent-form.o 2>&1 | FileCheck %s --check-prefix=BAD-PARENT-FORM --implicit-check-not=error:

# BAD-PARENT-FORM: error: bad-parent-form.o:(.debug_names): invalid form for DW_IDX_parent

# RUN: sed -E '/DW_IDX_die_offset/{n;s/[0-9]+.*DW_FORM_ref4/16/}' %S/Inputs/debug-names-a.s > bad-die-form.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-die-form.s -o bad-die-form.o
# RUN: not ld.lld --debug-names bad-die-form.o 2>&1 | FileCheck %s --check-prefix=BAD-DIE-FORM --implicit-check-not=error:

# BAD-DIE-FORM: error: bad-die-form.o:(.debug_names): unrecognized form encoding 16 in abbrev table

## Test errors in the entry pool.
# RUN: sed -E '/Lnames.:/{n;n;s/[0-9]+/3/}' %S/Inputs/debug-names-a.s > bad-abbrev-code.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-abbrev-code.s -o bad-abbrev-code.o
# RUN: not ld.lld --debug-names bad-abbrev-code.o 2>&1 | FileCheck %s --check-prefix=BAD-ABBREV-CODE --implicit-check-not=error:
# RUN: ld.lld --debug-names bad-abbrev-code.o -o bad-abbrev-code --noinhibit-exec
# RUN: llvm-dwarfdump --debug-names bad-abbrev-code | FileCheck %s --check-prefix=BAD-ABBREV-CODE-DWARF

# BAD-ABBREV-CODE: error: bad-abbrev-code.o:(.debug_names): abbrev code not found in abbrev table: 3

# BAD-ABBREV-CODE-DWARF:      Abbreviations [
# BAD-ABBREV-CODE-DWARF-NEXT:   Abbreviation 0x1 {
# BAD-ABBREV-CODE-DWARF-NEXT:     Tag: DW_TAG_subprogram
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_die_offset: DW_FORM_ref4
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_parent: DW_FORM_flag_present
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_compile_unit: DW_FORM_data1
# BAD-ABBREV-CODE-DWARF-NEXT:   }
# BAD-ABBREV-CODE-DWARF-NEXT:   Abbreviation 0x2 {
# BAD-ABBREV-CODE-DWARF-NEXT:     Tag: DW_TAG_structure_type
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_die_offset: DW_FORM_ref4
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_parent: DW_FORM_flag_present
# BAD-ABBREV-CODE-DWARF-NEXT:     DW_IDX_compile_unit: DW_FORM_data1
# BAD-ABBREV-CODE-DWARF-NEXT:   }
# BAD-ABBREV-CODE-DWARF-NEXT: ]
# BAD-ABBREV-CODE-DWARF:      Bucket 0
# BAD-ABBREV-CODE-DWARF-NOT:  Entry

# RUN: sed -E '/Lnames0:/{n;n;n;s/.*DW_IDX_die_offset.*//}' %S/Inputs/debug-names-a.s > missing-die-offset0.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 missing-die-offset0.s -o missing-die-offset0.o
# RUN: ld.lld --debug-names missing-die-offset0.o --noinhibit-exec -o missing-die-offset0 2>&1 | FileCheck %s --check-prefix=MISSING-DIE-OFFSET0

# MISSING-DIE-OFFSET0: warning: missing-die-offset0.o:(.debug_names): index entry is out of bounds

# RUN: sed -E '/Lnames1:/{n;n;n;s/.*DW_IDX_die_offset.*//}' %S/Inputs/debug-names-a.s > missing-die-offset1.s
# RUN: llvm-mc -filetype=obj -triple=x86_64 missing-die-offset1.s -o missing-die-offset1.o
# RUN: ld.lld --debug-names missing-die-offset1.o -o missing-die-offset1
# RUN: llvm-dwarfdump --debug-names missing-die-offset1 | FileCheck %s --check-prefix=MISSING-DIE-OFFSET1

# MISSING-DIE-OFFSET1:      Bucket 0 [
# MISSING-DIE-OFFSET1-NEXT:   Name 1 {
# MISSING-DIE-OFFSET1-NEXT:     Hash: 0x59796A
# MISSING-DIE-OFFSET1-NEXT:     String: {{.*}} "t1"
# MISSING-DIE-OFFSET1-NEXT:     Entry @ 0x65 {
# MISSING-DIE-OFFSET1-NEXT:       Abbrev: 0x2
# MISSING-DIE-OFFSET1-NEXT:       Tag: DW_TAG_structure_type
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_die_offset: 0x00230200
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_parent: <parent not indexed>
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_compile_unit: 0x00
# MISSING-DIE-OFFSET1-NEXT:     }
# MISSING-DIE-OFFSET1-NEXT:   }
# MISSING-DIE-OFFSET1-NEXT:   Name 2 {
# MISSING-DIE-OFFSET1-NEXT:     Hash: 0xEDDB6232
# MISSING-DIE-OFFSET1-NEXT:     String: {{.*}} "_start"
# MISSING-DIE-OFFSET1-NEXT:     Entry @ 0x6c {
# MISSING-DIE-OFFSET1-NEXT:       Abbrev: 0x1
# MISSING-DIE-OFFSET1-NEXT:       Tag: DW_TAG_subprogram
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_die_offset: 0x00000023
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_parent: <parent not indexed>
# MISSING-DIE-OFFSET1-NEXT:       DW_IDX_compile_unit: 0x00
# MISSING-DIE-OFFSET1-NEXT:     }
# MISSING-DIE-OFFSET1-NEXT:   }
# MISSING-DIE-OFFSET1-NEXT: ]
