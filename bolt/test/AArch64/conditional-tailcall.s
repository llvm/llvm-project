# Bolt cannot handle conditional tail calls.

# Example with CBZ:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym CBZ=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-CBZ

# Example with CBNZ:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym CBNZ=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-CBNZ

# Example with TBZ:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym TBZ=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-TBZ

# Example with TBNZ:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym TBNZ=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-TBNZ

# Example with B.EQ:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym BEQ=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BEQ

# Example with B.NE:
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o \
# RUN: --defsym BNE=1
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not --crash llvm-bolt %t.exe -o %t.bolt --skip-funcs foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BNE

# CHECK-CBZ:  FKI.TargetOffset == 0 && "0-bit relocation offset expected
# CHECK-CBNZ: FKI.TargetOffset == 0 && "0-bit relocation offset expected
# CHECK-TBZ:  FKI.TargetOffset == 0 && "0-bit relocation offset expected
# CHECK-TBNZ: FKI.TargetOffset == 0 && "0-bit relocation offset expected
# CHECK-BEQ:  FKI.TargetOffset == 0 && "0-bit relocation offset expected
# CHECK-BNE:  FKI.TargetOffset == 0 && "0-bit relocation offset expected

  .text
  .globl foo
  .type foo, %function
foo:
  .cfi_startproc

.ifdef CBZ
  cbz xzr, bar
.endif

.ifdef CBNZ
  cbnz xzr, bar
.endif

.ifdef TBZ
  tbz xzr, #0, bar
.endif

.ifdef TBNZ
  tbnz xzr, #0, bar
.endif

.ifdef BEQ
  cmp wzr, wzr
  b.eq bar
.endif

.ifdef BNE
  cmp wzr, wzr
  b.ne bar
.endif

  .cfi_endproc
.size foo, .-foo

  .globl bar
  .type bar, %function
bar:
  .cfi_startproc
  ret  xzr
  .cfi_endproc
.size bar, .-bar
