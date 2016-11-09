## mips-invalid-options-descriptor.elf has option descriptor in
## .MIPS.options with size of zero.
# RUN: not ld.lld %p/Inputs/mips-invalid-options-descriptor.elf -o %t2 2>&1 | \
# RUN:   FileCheck %s
# CHECK: error: Invalid data was encountered while parsing the file
