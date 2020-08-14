# REQUIRES: x86

## NOTE: Here we check that the first non-header section -- __text -- appears
## *exactly* `-headerpad` bytes from the end of the header. ld64 actually
## starts laying out the non-header sections in the __TEXT segment from the end
## of the (page-aligned) segment rather than the front, so its binaries
## typically have more than `-headerpad` bytes of actual padding. `-headerpad`
## just enforces a lower bound. We should consider implementing the same
## alignment behavior.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o -headerpad 0
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s --check-prefix=PAD0
# PAD0:      magic        cputype  cpusubtype  caps    filetype ncmds sizeofcmds               flags
# PAD0-NEXT: MH_MAGIC_64  X86_64   ALL         LIB64   EXECUTE  9     [[#%u, CMDSIZE:]] {{.*}}
# PAD0:      sectname __text
# PAD0-NEXT: segname __TEXT
# PAD0-NEXT: addr
# PAD0-NEXT: size
# PAD0-NEXT: offset [[#%u, CMDSIZE + 32]]

# RUN: lld -flavor darwinnew -o %t %t.o -headerpad 11
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s --check-prefix=PAD11
# PAD11:      magic        cputype  cpusubtype  caps    filetype ncmds sizeofcmds               flags
# PAD11-NEXT: MH_MAGIC_64  X86_64   ALL         LIB64   EXECUTE  9     [[#%u, CMDSIZE:]] {{.*}}
# PAD11:      sectname __text
# PAD11-NEXT: segname __TEXT
# PAD11-NEXT: addr
# PAD11-NEXT: size
# PAD11-NEXT: offset [[#%u, CMDSIZE + 32 + 0x11]]

.globl _main
_main:
  ret
