# REQUIRES: x86
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o

# RUN: not ld.lld a.o --fix-cortex-a8 --fix-cortex-a53-843419 2>&1 | FileCheck %s --check-prefix=ERR-ARM
# ERR-ARM: error: --fix-cortex-a8 is only supported on ARM targets
# ERR-ARM: error: --fix-cortex-a53-843419 is only supported on AArch64

# RUN: not ld.lld a.o --be8 2>&1 | FileCheck %s --check-prefix=ERR-BE8
# ERR-BE8: error: --be8 is only supported on ARM targets

# RUN: not ld.lld a.o --execute-only -z pac-plt -z force-bti -z bti-report=error \
# RUN:   -z pauth-report=error -z gcs-report=error -z gcs-report-dynamic=error \
# RUN:   -z gcs=always 2>&1 | FileCheck %s --check-prefix=CHECK-AARCH64
# CHECK-AARCH64:      error: --execute-only is only supported on AArch64 targets
# CHECK-AARCH64-NEXT: error: -z pac-plt only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z force-bti only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z bti-report only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z pauth-report only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z gcs-report only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z gcs-report-dynamic only supported on AArch64
# CHECK-AARCH64-NEXT: error: -z gcs only supported on AArch64

# RUN: not ld.lld a.o --relax-gp -z zicfilp-unlabeled-report=warning \
# RUN:   -z zicfilp-func-sig-report=warning -z zicfiss-report=warning \
# RUN:   -z zicfilp=unlabeled -z zicfiss=always 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-RISCV
# ERR-RISCV:      error: --relax-gp is only supported on RISC-V targets
# ERR-RISCV-NEXT: error: -z zicfilip-unlabeled-report is only supported on RISC-V targets
# ERR-RISCV-NEXT: error: -z zicfilip-func-sig-report is only supported on RISC-V targets
# ERR-RISCV-NEXT: error: -z zicfiss-report is only supported on RISC-V targets
# ERR-RISCV-NEXT: error: -z zicfilp is only supported on RISC-V targets
# ERR-RISCV-NEXT: error: -z zicfiss is only supported on RISC-V targets

# RUN: not ld.lld a.o --toc-optimize --pcrel-optimize 2>&1 | FileCheck %s --check-prefix=ERR-PPC64
# ERR-PPC64: error: --toc-optimize is only supported on PowerPC64 targets
# ERR-PPC64: error: --pcrel-optimize is only supported on PowerPC64 targets

# RUN: not ld.lld a.o -z execute-only-report=warning 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-EXECUTE-ONLY
# RUN: not ld.lld a.o -z execute-only-report=error 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-EXECUTE-ONLY
# ERR-EXECUTE-ONLY: error: -z execute-only-report only supported on AArch64 and ARM

# RUN: not ld.lld a.o -z execute-only-report=foo 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-EXECUTE-ONLY-INVALID
# ERR-EXECUTE-ONLY-INVALID: error: unknown -z execute-only-report= value: foo

.globl _start
_start:
