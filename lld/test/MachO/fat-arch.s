# REQUIRES: x86,aarch64
## FIXME: The tests doesn't run on windows right now because of llvm-mc (can't produce triple=arm64-apple-macos11.0)
# UNSUPPORTED: system-windows

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
# RUN: %lld %t.x86_64.o -bundle_loader %t.fat.exec.out -bundle -o %t.fat.bundle

# RUN: llvm-otool -h %t.fat.exec.out %t.fat.bundle| FileCheck %s --check-prefix=CPU-SUB
# CPU-SUB:            magic     cputype      cpusubtype   caps      filetype   ncmds sizeofcmds      flags
# CPU-SUB-NEXT:  0xfeedfacf   [[#CPU:]]   [[#CPU_SUB:]]   0x{{.+}}    {{.+}}  {{.+}}    {{.+}}      {{.+}}
# CPU-SUB-NEXT: Mach header
# CPU-SUB-NEXT:       magic     cputype       cpusubtype       caps    filetype ncmds sizeofcmds      flags
# CPU-SUB-NEXT:  0xfeedfacf    [[#CPU]]     [[#CPU_SUB]]   0x{{.+}}    {{.+}}  {{.+}}    {{.+}}      {{.+}}

.text
.global _main
_main:
  ret
