# REQUIRES: mips
# Check deducing MIPS specific ELF header flags from `emulation`.

# RUN: echo -n "BLOB" > %t.binary
# RUN: ld.lld -m elf32btsmip -r -b binary %t.binary -o %t.out
# RUN: llvm-readobj -h %t.out | FileCheck -check-prefix=O32 %s

# RUN: echo -n "BLOB" > %t.binary
# RUN: ld.lld -m elf32btsmipn32 -r -b binary %t.binary -o %t.out
# RUN: llvm-readobj -h %t.out | FileCheck -check-prefix=N32 %s

# RUN: echo -n "BLOB" > %t.binary
# RUN: ld.lld -m elf64btsmip -r -b binary %t.binary -o %t.out
# RUN: llvm-readobj -h %t.out | FileCheck -check-prefix=N64 %s

# O32:      Flags [
# O32-NEXT:   EF_MIPS_ABI_O32
# O32-NEXT: ]

# N32:      Flags [
# N32-NEXT:   EF_MIPS_ABI2
# N32-NEXT: ]

# N64:      Flags [
# N64-NEXT: ]
