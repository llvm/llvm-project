# REQUIRES: mips
# Check MIPS N64 ABI GOT relocations

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %p/Inputs/mips-pic.s -o %t.so.o
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.exe.o
# RUN: echo "SECTIONS { . = 0x30000; .text : { *(.text) } }" > %t.script
# RUN: ld.lld %t.so.o -shared -soname=t.so -o %t.so
# RUN: ld.lld %t.exe.o --script %t.script %t.so -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d -t --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readelf -r -s -A %t.exe | FileCheck -check-prefix=GOT %s

# CHECK: {{[0-9a-f]+}}1c8 l .text  0000000000000000 foo

# CHECK:      <__start>:
# CHECK-NEXT:    {{.*}}  ld      $2, -32736($gp)
# CHECK-NEXT:    {{.*}}  daddiu  $2,  $2, 456
# CHECK-NEXT:    {{.*}}  addiu   $2,  $2, -32704
# CHECK-NEXT:    {{.*}}  addiu   $2,  $2, -32720
# CHECK-NEXT:    {{.*}}  addiu   $2,  $2, -32712

# GOT: There are no relocations in this file.

# GOT: Symbol table '.symtab'
# GOT: {{.*}}: [[FOO:[0-9a-f]+]]     {{.*}} foo
# GOT: {{.*}}: [[GP:[0-9a-f]+]]      {{.*}} _gp
# GOT: {{.*}}: [[BAR:[0-9a-f]+]]     {{.*}} bar

# GOT:      Primary GOT:
# GOT-NEXT:  Canonical gp value: [[GP]]
# GOT-EMPTY:
# GOT-NEXT:  Reserved entries:
# GOT-NEXT:  Address     Access          Initial Purpose
# GOT-NEXT:   {{.*}} -32752(gp) 0000000000000000 Lazy resolver
# GOT-NEXT:   {{.*}} -32744(gp) 8000000000000000 Module pointer (GNU extension)
# GOT-EMPTY:
# GOT-NEXT:  Local entries:
# GOT-NEXT:  Address     Access          Initial
# GOT-NEXT:   {{.*}} -32736(gp) 0000000000030000
# GOT-NEXT:   {{.*}} -32728(gp) 0000000000040000
# GOT-NEXT:   {{.*}} -32720(gp) [[BAR]]
# GOT-NEXT:   {{.*}} -32712(gp) [[FOO]]
# GOT-EMPTY:
# GOT-NEXT:  Global entries:
# GOT-NEXT:  Address     Access          Initial         Sym.Val. Type Ndx Name
# GOT-NEXT:   {{.*}} -32704(gp) 0000000000000000 0000000000000000 FUNC UND foo1a

  .text
  .global  __start, bar
__start:
  ld      $v0,%got_page(foo)($gp)             # R_MIPS_GOT_PAGE
  daddiu  $v0,$v0,%got_ofst(foo)              # R_MIPS_GOT_OFST
  addiu   $v0,$v0,%got_disp(foo1a)            # R_MIPS_GOT_DISP
  addiu   $v0,$v0,%got_disp(bar)              # R_MIPS_GOT_DISP
  addiu   $v0,$v0,%got_disp(foo)              # R_MIPS_GOT_DISP

bar:
  nop
foo:
  nop
