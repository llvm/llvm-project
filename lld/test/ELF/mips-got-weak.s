# REQUIRES: mips
# Check R_MIPS_GOT16 relocation against weak symbols.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t1.so
# RUN: llvm-readelf -r --dyn-syms --dynamic-table -A %t1.so \
# RUN:   | FileCheck -check-prefixes=CHECK,NOSYM %s
# RUN: ld.lld %t.o -shared -Bsymbolic -o %t2.so
# RUN: llvm-readelf -r --dyn-syms --dynamic-table -A %t2.so \
# RUN:   | FileCheck -check-prefixes=CHECK,SYM %s

# CHECK: There are no relocations in this file.

# CHECK: Symbol table '.dynsym'
# CHECK-DAG: [[FOO:[0-9a-f]+]]     0 NOTYPE  WEAK   DEFAULT    8 foo
# CHECK-DAG:          00000000     0 NOTYPE  WEAK   DEFAULT  UND bar
# CHECK-DAG: [[SYM:[0-9a-f]+]]     0 NOTYPE  GLOBAL DEFAULT    8 sym

# CHECK: Dynamic section
# CHECK: (MIPS_SYMTABNO)      4
# NOSYM: (MIPS_LOCAL_GOTNO)   2
# NOSYM: (MIPS_GOTSYM)        0x1
#   SYM: (MIPS_LOCAL_GOTNO)   4
#   SYM: (MIPS_GOTSYM)        0x3

# NOSYM:      Primary GOT:
# NOSYM-NOT:   Local entries:
# NOSYM:       Global entries:
# NOSYM-NEXT:       Access  Initial Sym.Val. Type    Ndx Name
# NOSYM-NEXT:   -32744(gp)  [[FOO]]  [[FOO]] NOTYPE    8 foo
# NOSYM-NEXT:   -32740(gp) 00000000 00000000 NOTYPE  UND bar
# NOSYM-NEXT:   -32736(gp)  [[SYM]]  [[SYM]] NOTYPE    8 sym

# SYM:      Primary GOT:
# SYM:       Local entries:
# SYM-NEXT:       Access  Initial
# SYM-NEXT:   -32744(gp)  [[FOO]]
# SYM-NEXT:   -32740(gp)  [[SYM]]
# SYM:       Global entries:
# SYM-NEXT:       Access  Initial Sym.Val. Type    Ndx Name
# SYM-NEXT:   -32736(gp) 00000000 00000000 NOTYPE  UND bar

  .text
  .global  sym
  .weak    foo,bar
func:
  lw      $t0,%got(foo)($gp)
  lw      $t0,%got(bar)($gp)
  lw      $t0,%got(sym)($gp)

  .data
  .weak foo
foo:
  .word 0
sym:
  .word 0
