# REQUIRES: x86

## NOTE: Here we check that the first non-header section -- __text -- appears
## *exactly* `-headerpad` bytes from the end of the header. ld64 actually
## starts laying out the non-header sections in the __TEXT segment from the end
## of the (page-aligned) segment rather than the front, so its binaries
## typically have more than `-headerpad` bytes of actual padding. `-headerpad`
## just enforces a lower bound. We should consider implementing the same
## alignment behavior.

# RUN: rm -rf %t; mkdir -p %t

################ Check default behavior
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test %t/test.o
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PADx
#
# PADx:      magic        {{.+}}  ncmds  sizeofcmds         flags
# PADx-NEXT: MH_MAGIC_64  {{.+}}  [[#]]  [[#%u, CMDSIZE:]]  {{.*}}
# PADx:      sectname __text
# PADx-NEXT: segname __TEXT
# PADx-NEXT: addr
# PADx-NEXT: size
# PADx-NEXT: offset [[#%u, CMDSIZE + 0x20 + 0x20]]

################ Zero pad, no LCDylibs
# RUN: %lld -o %t/test %t/test.o -headerpad 0
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PAD0
# RUN: %lld -o %t/test %t/test.o -headerpad 0 -headerpad_max_install_names
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PAD0
#
# PAD0:      magic        {{.+}}  ncmds  sizeofcmds         flags
# PAD0-NEXT: MH_MAGIC_64  {{.+}}  [[#]]  [[#%u, CMDSIZE:]]  {{.*}}
# PAD0:      sectname __text
# PAD0-NEXT: segname __TEXT
# PAD0-NEXT: addr
# PAD0-NEXT: size
# PAD0-NEXT: offset [[#%u, CMDSIZE + 0x20 + 0]]

################ Each lexical form of a hex number, no LCDylibs
# RUN: %lld -o %t/test %t/test.o -headerpad 11
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PAD11
# RUN: %lld -o %t/test %t/test.o -headerpad 0x11
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PAD11
# RUN: %lld -o %t/test %t/test.o -headerpad 0X11 -headerpad_max_install_names
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=PAD11
#
# PAD11:      magic        {{.+}}  ncmds  sizeofcmds         flags
# PAD11-NEXT: MH_MAGIC_64  {{.+}}  [[#]]  [[#%u, CMDSIZE:]]  {{.*}}
# PAD11:      sectname __text
# PAD11-NEXT: segname __TEXT
# PAD11-NEXT: addr
# PAD11-NEXT: size
# PAD11-NEXT: offset [[#%u, CMDSIZE + 0x20 + 0x11]]

################ Each & all 3 kinds of LCDylib
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/null.o
# RUN: %lld -o %t/libnull.dylib %t/null.o -dylib \
# RUN:     -headerpad_max_install_names
# RUN: llvm-objdump --macho --all-headers %t/libnull.dylib | FileCheck %s --check-prefix=PADMAX
# RUN: %lld -o %t/libnull.dylib %t/null.o -dylib \
# RUN:     -headerpad_max_install_names -lSystem
# RUN: llvm-objdump --macho --all-headers %t/libnull.dylib | FileCheck %s --check-prefix=PADMAX
# RUN: %lld -o %t/libnull.dylib %t/null.o -dylib \
# RUN:     -headerpad_max_install_names \
# RUN:     -lSystem -sub_library libSystem
# RUN: llvm-objdump --macho --all-headers %t/libnull.dylib | FileCheck %s --check-prefix=PADMAX
#
# PADMAX:      magic        {{.+}}  ncmds        sizeofcmds         flags
# PADMAX-NEXT: MH_MAGIC_64  {{.+}}  [[#%u, N:]]  [[#%u, CMDSIZE:]]  {{.*}}
# PADMAX:      sectname __text
# PADMAX-NEXT: segname __TEXT
# PADMAX-NEXT: addr
# PADMAX-NEXT: size
# PADMAX-NEXT: offset [[#%u, CMDSIZE + 0x20 + mul(0x400, N - 9)]]

################ All 3 kinds of LCDylib swamped by a larger override
# RUN: %lld -o %t/libnull.dylib %t/null.o -dylib \
# RUN:     -headerpad_max_install_names -headerpad 0x1001 \
# RUN:     -lSystem -sub_library libSystem
# RUN: llvm-objdump --macho --all-headers %t/libnull.dylib | FileCheck %s --check-prefix=PADOVR
#
# PADOVR:      magic        {{.+}}  ncmds        sizeofcmds         flags
# PADOVR-NEXT: MH_MAGIC_64  {{.+}}  [[#%u, N:]]  [[#%u, CMDSIZE:]]  {{.*}}
# PADOVR:      sectname __text
# PADOVR-NEXT: segname __TEXT
# PADOVR-NEXT: addr
# PADOVR-NEXT: size
# PADOVR-NEXT: offset [[#%u, CMDSIZE + 0x20 + 0x1001]]

.globl _main
_main:
  ret
