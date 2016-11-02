## Spec says that "If a file has no section header table, e_shnum holds the value zero.", though
## in this test case it holds non-zero and lld may crash.
# RUN: not ld.lld %p/Inputs/invalid-e_shnum.elf -o %t2 2>&1 | FileCheck %s
# CHECK: Invalid data was encountered while parsing the file
