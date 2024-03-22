# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux asm -o a.o

## Default case: abc and abx included in text.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .abx) } }" > a.t
# RUN: ld.lld -o out --script a.t a.o
# RUN: llvm-objdump --section-headers out | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s
# SEC-DEFAULT:      Sections:
# SEC-DEFAULT-NEXT: Idx Name          Size
# SEC-DEFAULT-NEXT:   0               00000000
# SEC-DEFAULT-NEXT:   1 .text         00000008
# SEC-DEFAULT-NEXT:   2 .abcd         00000004
# SEC-DEFAULT-NEXT:   3 .ad           00000004
# SEC-DEFAULT-NEXT:   4 .ag           00000004
# SEC-DEFAULT-NEXT:   5 .comment      00000008 {{[0-9a-f]*}}
# SEC-DEFAULT-NEXT:   6 .symtab       00000030
# SEC-DEFAULT-NEXT:   7 .shstrtab     00000038
# SEC-DEFAULT-NEXT:   8 .strtab       00000008

## Now replace the symbol with '?' and check that results are the same.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .ab?) } }" > b.t
# RUN: ld.lld -o out -T b.t a.o
# RUN: llvm-objdump --section-headers out | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s

## Now see how replacing '?' with '*' will consume whole abcd.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .ab*) } }" > c.t
# RUN: ld.lld -o out --script c.t a.o
# RUN: llvm-objdump --section-headers out | \
# RUN:   FileCheck -check-prefix=SEC-ALL %s
# SEC-ALL:      Sections:
# SEC-ALL-NEXT: Idx Name          Size
# SEC-ALL-NEXT:   0               00000000
# SEC-ALL-NEXT:   1 .text         0000000c
# SEC-ALL-NEXT:   2 .ad           00000004
# SEC-ALL-NEXT:   3 .ag           00000004
# SEC-ALL-NEXT:   4 .comment      00000008
# SEC-ALL-NEXT:   5 .symtab       00000030
# SEC-ALL-NEXT:   6 .shstrtab     00000032
# SEC-ALL-NEXT:   7 .strtab       00000008

## All sections started with .a are merged.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.a*) } }" > d.t
# RUN: ld.lld -o out --script d.t a.o
# RUN: llvm-objdump --section-headers out | \
# RUN:   FileCheck -check-prefix=SEC-NO %s
# SEC-NO: Sections:
# SEC-NO-NEXT: Idx Name          Size
# SEC-NO-NEXT:   0               00000000
# SEC-NO-NEXT:   1 .text         00000014
# SEC-NO-NEXT:   2 .comment      00000008
# SEC-NO-NEXT:   3 .symtab       00000030
# SEC-NO-NEXT:   4 .shstrtab     0000002a
# SEC-NO-NEXT:   5 .strtab       00000008

#--- asm
.text
.section .abc,"ax",@progbits
.long 0

.text
.section .abx,"ax",@progbits
.long 0

.text
.section .abcd,"ax",@progbits
.long 0

.text
.section .ad,"ax",@progbits
.long 0

.text
.section .ag,"ax",@progbits
.long 0


.globl _start
_start:

#--- bracket.lds
# RUN: ld.lld -T bracket.lds a.o -o out
# RUN: llvm-objdump --section-headers out | FileCheck %s --check-prefix=SEC-DEFAULT
SECTIONS {
  .text : { *([.]abc .ab[v-y] ) }
}

## Test a few non-wildcard characters rejected by GNU ld.

#--- lbrace.lds
# RUN: not ld.lld -T lbrace.lds a.o 2>&1 | FileCheck %s --check-prefix=ERR-LBRACE --match-full-lines --strict-whitespace
#      ERR-LBRACE:{{.*}}: section pattern is expected
# ERR-LBRACE-NEXT:>>>   .text : { *(.a* { ) }
# ERR-LBRACE-NEXT:>>>                   ^
SECTIONS {
  .text : { *(.a* { ) }
}

#--- lbrace2.lds
# RUN: not ld.lld -T lbrace2.lds a.o 2>&1 | FileCheck %s --check-prefix=ERR-LBRACE2 --match-full-lines --strict-whitespace
#      ERR-LBRACE2:{{.*}}: section pattern is expected
# ERR-LBRACE2-NEXT:>>>   .text : { *(.a*{) }
# ERR-LBRACE2-NEXT:>>>                  ^
SECTIONS {
  .text : { *(.a*{) }
}

#--- lparen.lds
# RUN: not ld.lld -T lparen.lds a.o 2>&1 | FileCheck %s --check-prefix=ERR-LPAREN --match-full-lines --strict-whitespace
#      ERR-LPAREN:{{.*}}: section pattern is expected
# ERR-LPAREN-NEXT:>>>   .text : { *(.a* ( ) }
# ERR-LPAREN-NEXT:>>>                   ^
SECTIONS {
  .text : { *(.a* ( ) }
}

#--- rbrace.lds
# RUN: not ld.lld -T rbrace.lds a.o 2>&1 | FileCheck %s --check-prefix=ERR-RBRACE --match-full-lines --strict-whitespace
#      ERR-RBRACE:{{.*}}: section pattern is expected
# ERR-RBRACE-NEXT:>>>   .text : { *(.a* x = 3; } ) }
# ERR-RBRACE-NEXT:>>>                          ^
SECTIONS {
  .text : { *(.a* x = 3; } ) }
}

#--- rparen.lds
# RUN: not ld.lld -T rparen.lds a.o 2>&1 | FileCheck %s --check-prefix=ERR-RPAREN --match-full-lines --strict-whitespace
#      ERR-RPAREN:{{.*}}: expected filename pattern
# ERR-RPAREN-NEXT:>>>   .text : { *(.a* ) ) }
# ERR-RPAREN-NEXT:>>>                     ^
SECTIONS {
  .text : { *(.a* ) ) }
}
