# REQUIRES: aarch64, x86
# RUN: rm -rf %t; mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t/watchos-test.o

# RUN: %lld -lSystem -o %t/test %t/test.o
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=NO-ENCRYPTION -DSUFFIX=_64

# RUN: %lld -lSystem -encryptable -o %t/test %t/test.o
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=ENCRYPTION -DSUFFIX=_64 -D#PAGE_SIZE=4096

# RUN: %lld-watchos -lSystem -o %t/watchos-test %t/watchos-test.o
# RUN: llvm-objdump --macho --all-headers %t/watchos-test | FileCheck %s --check-prefix=ENCRYPTION -DSUFFIX= -D#PAGE_SIZE=16384

# RUN: %lld-watchos -lSystem -no_encryption -o %t/watchos-test %t/watchos-test.o
# RUN: llvm-objdump --macho --all-headers %t/watchos-test | FileCheck %s --check-prefix=NO-ENCRYPTION -DSUFFIX=

# ENCRYPTION:      segname __TEXT
# ENCRYPTION-NEXT: vmaddr
# ENCRYPTION-NEXT: vmsize
# ENCRYPTION-NEXT: fileoff 0
# ENCRYPTION-NEXT: filesize [[#TEXT_SIZE:]]

# ENCRYPTION:      cmd LC_ENCRYPTION_INFO[[SUFFIX]]{{$}}
# ENCRYPTION-NEXT: cmdsize
# ENCRYPTION-NEXT: cryptoff [[#PAGE_SIZE]]
# ENCRYPTION-NEXT: cryptsize [[#TEXT_SIZE - PAGE_SIZE]]
# ENCRYPTION-NEXT: cryptid 0

# NO-ENCRYPTION-NOT: LC_ENCRYPTION_INFO[[SUFFIX]]{{$}}

.globl _main
.p2align 2
_main:
  ret
