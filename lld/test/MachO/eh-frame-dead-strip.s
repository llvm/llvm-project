# REQUIRES: x86, aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 %t/strong.s -o %t/strong_x86_64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11.0 %t/weak.s -o %t/weak_x86_64.o
# RUN: %lld -dylib -dead_strip %t/strong_x86_64.o %t/weak_x86_64.o -o %t/libstrongweak_x86_64.dylib
# RUN: llvm-dwarfdump --eh-frame %t/libstrongweak_x86_64.dylib | FileCheck --check-prefixes CHECK,X86_64 %s
# RUN: %lld -dylib -dead_strip %t/weak_x86_64.o %t/strong_x86_64.o -o %t/libweakstrong_x86_64.dylib
# RUN: llvm-dwarfdump --eh-frame %t/libweakstrong_x86_64.dylib | FileCheck --check-prefixes CHECK,X86_64 %s

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/strong.s -o %t/strong_arm64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/weak.s -o %t/weak_arm64.o
# RUN: %lld -arch arm64 -dylib -dead_strip %t/strong_arm64.o %t/weak_arm64.o -o %t/libstrongweak_arm64.dylib
# RUN: llvm-dwarfdump --eh-frame %t/libstrongweak_arm64.dylib | FileCheck --check-prefixes CHECK,ARM64 %s
# RUN: %lld -arch arm64 -dylib -dead_strip %t/weak_arm64.o %t/strong_arm64.o -o %t/libweakstrong_arm64.dylib
# RUN: llvm-dwarfdump --eh-frame %t/libweakstrong_arm64.dylib | FileCheck --check-prefixes CHECK,ARM64 %s

## Verify that unneeded FDEs (and their CIEs) are dead-stripped even if they
## point to a live symbol (e.g. because we had multiple weak definitions).

# CHECK: .eh_frame contents:
# X86_64: 00000000 00000014 00000000 CIE
# X86_64: 00000018 0000001c 0000001c FDE cie=00000000
# ARM64: 00000000 00000010 00000000 CIE
# ARM64: 00000014 00000018 00000018 FDE cie=00000000
# CHECK-NOT: CIE
# CHECK-NOT: FDE

#--- strong.s
.globl _fun
_fun:
  .cfi_startproc
  ## cfi_escape cannot be encoded in compact unwind
  .cfi_escape 0
  ret
  .cfi_endproc

#--- weak.s
.globl _fun
.weak_definition _fun
_fun:
  .cfi_startproc
  ## cfi_escape cannot be encoded in compact unwind
  .cfi_escape 0
  ret
  .cfi_endproc
