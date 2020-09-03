# REQUIRES: mips
# Check R_MIPS_CALL_HI16 / R_MIPS_CALL_LO16 relocations calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s
# RUN: llvm-readelf -r -s -A %t.so | FileCheck -check-prefix=GOT %s

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <foo>:
# CHECK-NEXT:    {{.*}}:  lui  $2, 0
# CHECK-NEXT:    {{.*}}:  lw   $2, -32736($2)
# CHECK-NEXT:    {{.*}}:  lui  $2, 0
# CHECK-NEXT:    {{.*}}:  lw   $2, -32744($2)
# CHECK-NEXT:    {{.*}}:  lui  $2, 0
# CHECK-NEXT:    {{.*}}:  lw   $2, -32740($2)

# GOT: There are no relocations in this file.

# GOT: Symbol table '.symtab'
# GOT: {{.*}}:  [[LOC1:[0-9a-f]+]]  {{.*}} loc1
# GOT: {{.*}}:  [[LOC2:[0-9a-f]+]]  {{.*}} loc2

# GOT:      Primary GOT:
# GOT-NEXT:  Canonical gp value:
# GOT-EMPTY:
# GOT-NEXT:  Reserved entries:
# GOT-NEXT:    Address     Access  Initial Purpose
# GOT-NEXT:   {{.*}} -32752(gp) 00000000 Lazy resolver
# GOT-NEXT:   {{.*}} -32748(gp) 80000000 Module pointer (GNU extension)
# GOT-EMPTY:
# GOT-NEXT:  Local entries:
# GOT-NEXT:    Address     Access  Initial
# GOT-NEXT:   {{.*}} -32744(gp) [[LOC1]]
# GOT-NEXT:   {{.*}} -32740(gp) [[LOC2]]
# GOT-EMPTY:
# GOT-NEXT:  Global entries:
# GOT-NEXT:    Address     Access  Initial Sym.Val. Type    Ndx Name
# GOT-NEXT:   {{.*}} -32736(gp) 00000000 00000000 NOTYPE  UND bar

  .text
  .global foo
foo:
  lui   $2, %call_hi(bar)
  lw    $2, %call_lo(bar)($2)
  lui   $2, %call_hi(loc1)
  lw    $2, %call_lo(loc1)($2)
  lui   $2, %call_hi(loc2)
  lw    $2, %call_lo(loc2)($2)
loc1:
  nop
loc2:
  nop
