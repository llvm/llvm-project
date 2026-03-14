# REQUIRES: x86,aarch64
## FIXME: The tests doesn't run on windows right now because of llvm-mc (can't produce triple=arm64-apple-macos11.0)
# UNSUPPORTED: system-windows

# RUN: llvm-mc -filetype=obj -triple=i386-apple-darwin %s -o %t.i386.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %s -o %t.arm64.o

# RUN: llvm-lipo %t.i386.o %t.x86_64.o -create -o %t.fat.o
# RUN: %lld -o /dev/null %t.fat.o
# RUN: llvm-lipo %t.i386.o -create -o %t.noarch.o
# RUN: not %no-fatal-warnings-lld -o /dev/null %t.noarch.o 2>&1 | \
# RUN:    FileCheck %s -DFILE=%t.noarch.o
# CHECK: warning: [[FILE]]: ignoring file because it is universal (i386) but does not contain the x86_64 architecture

# RUN: not %lld -arch arm64 -o /dev/null %t.fat.o 2>&1 | \
# RUN:    FileCheck --check-prefix=CHECK-FAT %s -DFILE=%t.fat.o
# CHECK-FAT: error: [[FILE]]: ignoring file because it is universal (i386,x86_64) but does not contain the arm64 architecture

## Validates that we read the cpu-subtype correctly from a fat exec.
# RUN: %lld -o %t.x86_64.out %t.x86_64.o
# RUN: %lld -arch arm64 -o %t.arm64.out %t.arm64.o
# RUN: llvm-lipo %t.x86_64.out %t.arm64.out -create -o %t.fat.exec.out
# RUN: %lld -arch x86_64 %t.x86_64.o -bundle_loader %t.fat.exec.out -bundle -o %t.fat.bundle

# RUN: llvm-otool -h %t.fat.bundle > %t.bundle_header.txt
# RUN: llvm-otool -f %t.fat.exec.out >> %t.bundle_header.txt
# RUN: cat %t.bundle_header.txt | FileCheck %s --check-prefix=CPU-SUB

# CPU-SUB:            magic     cputype      cpusubtype   caps      filetype   ncmds sizeofcmds      flags
# CPU-SUB-NEXT:  0xfeedfacf     16777223              3  0x{{.+}}    {{.+}}  {{.+}}    {{.+}}      {{.+}}

# CPU-SUB: Fat headers
# CPU-SUB: nfat_arch 2
# CPU-SUB: architecture 0
# CPU-SUB-NEXT:    cputype 16777223
# CPU-SUB-NEXT:    cpusubtype 3
# CPU-SUB: architecture 1
# CPU-SUB-NEXT:    cputype 16777228
# CPU-SUB-NEXT:    cpusubtype 0

.text
.global _main
_main:
  ret
