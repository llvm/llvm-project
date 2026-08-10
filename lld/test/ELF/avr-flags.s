; REQUIRES: avr

; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=avr5 %s -o %t-v5
; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=avrxmega3 %s -o %t-xmega3
; RUN: ld.lld %t-v5 -o %t-v5.out
; RUN: ld.lld %t-xmega3 -o %t-xmega3.out
; RUN: llvm-readobj -h %t-v5.out | FileCheck --check-prefix V5 %s
; RUN: llvm-readobj -h %t-xmega3.out | FileCheck --check-prefix XMEGA3 %s

;; Ensure LLD won't silently mix object files targeting different ISAs.
; RUN: not ld.lld %t-v5 %t-xmega3 -o /dev/null 2>&1 | FileCheck --check-prefix ERR %s
; ERR: error: {{.*}}: cannot link object files with incompatible target ISA

; V5:  Flags [ (0x85)
; V5:  EF_AVR_ARCH_AVR5 (0x5)
; V5:  EF_AVR_LINKRELAX_PREPARED (0x80)
; XMEGA3: Flags [ (0xE7)
; XMEGA3: EF_AVR_ARCH_XMEGA3 (0x67)
; XMEGA3: EF_AVR_LINKRELAX_PREPARED (0x80)
