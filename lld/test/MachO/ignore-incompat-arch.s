# REQUIRES: x86, aarch64
## Test that LLD correctly ignored archives with incompatible architecture without crashing.

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos  %t/callee.s -o %t/callee_arm64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos  %t/callee.s -o %t/callee_x86_64.o

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos  %t/caller.s -o %t/caller_arm64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos  %t/caller.s -o %t/caller_x86_64.o

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos  %t/main.s -o %t/main_arm64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos  %t/main.s -o %t/main_x86_64.o

# RUN: llvm-ar rc %t/libcallee_arm64.a %t/callee_arm64.o
# RUN: llvm-ar r %t/libcallee_x86.a %t/callee_x86_64.o

# RUN: llvm-ar r %t/libcaller_arm64.a %t/caller_arm64.o
# RUN: llvm-ar r %t/libcaller_x86.a %t/caller_x86_64.o

## Symbol from the arm64 archive should be ignored even tho it appears before the x86 archive.
# RUN: %no-fatal-warnings-lld -map %t/x86_a.map -arch x86_64 %t/main_x86_64.o %t/libcallee_arm64.a %t/libcallee_x86.a %t/libcaller_x86.a -o %t/x86_a.out 2>&1 \
# RUN:     | FileCheck -check-prefix=X86-WARNING %s

# RUN: %no-fatal-warnings-lld -map %t/x86_b.map -arch x86_64 %t/main_x86_64.o %t/libcallee_x86.a %t/libcallee_arm64.a %t/libcaller_x86.a -o %t/x86_b.out 2>&1 \
# RUN:     | FileCheck -check-prefix=X86-WARNING %s

# RUN: %no-fatal-warnings-lld -map %t/arm64_a.map -arch arm64 %t/main_arm64.o %t/libcallee_x86.a %t/libcallee_arm64.a %t/libcaller_arm64.a -o %t/arm64_a.out 2>&1 \
# RUN:     | FileCheck -check-prefix=ARM64-WARNING %s

# RUN: %no-fatal-warnings-lld -map %t/arm64_b.map -arch arm64 %t/main_arm64.o %t/libcallee_arm64.a %t/libcallee_x86.a %t/libcaller_arm64.a -o %t/arm64_b.out 2>&1 \
# RUN:     | FileCheck -check-prefix=ARM64-WARNING %s

## Verify that the output doesn't take any symbol from the in-compat archive
# RUN: FileCheck --check-prefix=SYM-X86 %s --input-file=%t/x86_a.map
# RUN: FileCheck --check-prefix=SYM-X86 %s --input-file=%t/x86_b.map

# RUN: FileCheck --check-prefix=SYM-ARM64 %s --input-file=%t/arm64_a.map
# RUN: FileCheck --check-prefix=SYM-ARM64 %s --input-file=%t/arm64_b.map


# X86-WARNING: libcallee_arm64.a has architecture arm64 which is incompatible with target architecture x86_64

# ARM64-WARNING: libcallee_x86.a has architecture x86_64 which is incompatible with target architecture arm64

# SYM-X86-NOT: libcallee_arm64.a
# SYM-X86: {{.+}}main_x86_64.o
# SYM-X86: {{.+}}libcallee_x86.a(callee_x86_64.o)
# SYM-X86: {{.+}}libcaller_x86.a(caller_x86_64.o)

# SYM-ARM64-NOT: libcallee_x86.a
# SYM-ARM64: {{.+}}main_arm64.o
# SYM-ARM64: {{.+}}libcallee_arm64.a(callee_arm64.o)
# SYM-ARM64: {{.+}}libcaller_arm64.a(caller_arm64.o)


#--- callee.s
.globl _callee
_callee:
  ret

#--- caller.s
.globl _caller
_caller:
  .quad _callee
  ret

#--- main.s
.globl _main
_main:
  .quad _caller
  ret
