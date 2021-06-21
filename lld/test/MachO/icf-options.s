# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o
# RUN: %lld -lSystem --icf=all -o %t/all %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-EMPTY --allow-empty
# RUN: %lld -lSystem --icf=none -o %t/none %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-EMPTY --allow-empty
# RUN: %lld -lSystem -no_deduplicate -o %t/no_dedup %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-EMPTY --allow-empty
# RUN: not %lld -lSystem --icf=safe -o %t/safe %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-SAFE
# RUN: not %lld -lSystem --icf=junk -o %t/junk %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-JUNK
# RUN: not %lld -lSystem --icf=all -no_deduplicate -o %t/clash %t/main.o 2>&1 \
# RUN:     | FileCheck %s --check-prefix=DIAG-CLASH

# DIAG-EMPTY-NOT: {{.}}
# DIAG-SAFE: `--icf=safe' is not yet implemented, reverting to `none'
# DIAG-JUNK: unknown --icf=OPTION `junk', defaulting to `none'
# DIAG-CLASH: `--icf=all' conflicts with -no_deduplicate, setting to `none'

# RUN: llvm-objdump -d --syms %t/all | FileCheck %s --check-prefix=FOLD
# RUN: llvm-objdump -d --syms %t/none | FileCheck %s --check-prefix=NOOP
# RUN: llvm-objdump -d --syms %t/no_dedup | FileCheck %s --check-prefix=NOOP

# FOLD-LABEL: SYMBOL TABLE:
# FOLD:       [[#%x,MAIN:]] g   F __TEXT,__text _main
# FOLD:       [[#%x,F:]]    g   F __TEXT,__text _f1
# FOLD:       [[#%x,F]]     g   F __TEXT,__text _f2

# FOLD-LABEL: Disassembly of section __TEXT,__text:
# FOLD:       [[#%x,MAIN]] <_main>:
# FOLD-NEXT:  callq 0x[[#%x,F]]  <_f2>
# FOLD-NEXT:  callq 0x[[#%x,F]]  <_f2>

# NOOP-LABEL: SYMBOL TABLE:
# NOOP:       [[#%x,MAIN:]] g   F __TEXT,__text _main
# NOOP:       [[#%x,F1:]]   g   F __TEXT,__text _f1
# NOOP:       [[#%x,F2:]]   g   F __TEXT,__text _f2

# NOOP-LABEL: Disassembly of section __TEXT,__text:
# NOOP:       [[#%x,MAIN]] <_main>:
# NOOP-NEXT:  callq 0x[[#%x,F1]]  <_f1>
# NOOP-NEXT:  callq 0x[[#%x,F2]]  <_f2>

.subsections_via_symbols
.text
.p2align 2

.globl _f1
_f1:
  movl $0, %eax
  ret

.globl _f2
_f2:
  movl $0, %eax
  ret

.globl _main
_main:
  callq _f1
  callq _f2
  ret
