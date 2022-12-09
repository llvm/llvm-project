# REQUIRES: x86, aarch64
# RUN: llvm-mc -filetype=obj -triple=i386-apple-darwin %s -o %t.i386.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %s -o %t.arm64.o

# RUN: llvm-lipo %t.i386.o %t.x86_64.o -create -o %t.fat.o
# RUN: %lld -o /dev/null %t.fat.o
# RUN: llvm-lipo %t.i386.o -create -o %t.noarch.o
# RUN: not %lld -o /dev/null %t.noarch.o 2>&1 | \
# RUN:    FileCheck %s -DFILE=%t.noarch.o
# CHECK: error: unable to find matching architecture in [[FILE]]

## Validates that we read the cpu-subtype correctly from a fat exec.
# RUN: %lld -o %t.x86_64.out %t.x86_64.o
# RUN: %lld -arch arm64 -o %t.arm64.out %t.arm64.o
# RUN: llvm-lipo %t.x86_64.out %t.arm64.out -create -o %t.fat.exec.out

# RUN: llvm-otool -h %t.fat.exec.out | FileCheck %s --check-prefix=PRE-COND
# PRE-COND:             magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
# PRE-COND-NEXT:  0xfeedfacf 16777223          3  0x80           2    13        648 0x00200085

# RUN: %lld %t.x86_64.o -bundle_loader %t.fat.exec.out -bundle -o %t.fat.bundle
# RUN: llvm-otool -h %t.fat.bundle | FileCheck %s --check-prefix=POST-COND
# POST-COND:            magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
# POST-COND-NEXT:  0xfeedfacf 16777223          3  0x00           8    10        520 0x00000085

.text
.global _main
_main:
  ret
