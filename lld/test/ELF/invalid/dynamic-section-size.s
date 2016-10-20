## dynamic-section-sh_size.elf has incorrect sh_size of dynamic section.
# RUN: not ld.lld %p/Inputs/dynamic-section-sh_size.elf -o %t2 2>&1 | \
# RUN:   FileCheck %s
# CHECK: getSectionContentsAsArray failed: Invalid data was encountered while parsing the file
