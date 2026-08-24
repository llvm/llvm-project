## Check that BOLT preserves a branch to code immediately past the size of a
## function. Such code can result from a temporary assembler symbol that is
## not retained in the linked binary.

# REQUIRES: system-linux, target=aarch64{{.*}}

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q,-e,_start
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs=1 --lite=0 2>&1 \
# RUN:   | FileCheck %s --check-prefix=WARNING
# RUN: llvm-objdump -d --no-show-raw-insn -j .text %t.bolt \
# RUN:   | FileCheck %s --check-prefix=DISASM

# WARNING: BOLT-WARNING: jump past end detected

# DISASM: <f>:
# DISASM-NEXT: {{.*}}cbnz x0,
# DISASM-NEXT: {{.*}}ret

  .text
  .global _start
  .type _start, %function
_start:
  b f
  .size _start, .-_start

  .global f
  .type f, %function
f:
  cbnz x0, .Lf_helper
  ret
  .size f, .-f

  .type .Lf_helper, %function
.Lf_helper:
  mov x1, #42
  ret
  .size .Lf_helper, .-.Lf_helper

  .type next, %function
next:
  ret
  .size next, .-next
