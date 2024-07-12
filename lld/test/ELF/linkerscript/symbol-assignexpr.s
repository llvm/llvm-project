# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo "SECTIONS { symbol2 = symbol; }" > %t2.script
# RUN: not ld.lld -o /dev/null -T %t2.script %t.o -Map=%t.map 2>&1 \
# RUN:   | FileCheck --check-prefix=ERR %s --implicit-check-not=error:
# RUN: FileCheck --input-file=%t.map %s --check-prefix=MAP
# RUN: not ld.lld -o /dev/null --noinhibit-exec -T %t2.script %t.o 2>&1 \
# RUN:   | FileCheck --check-prefix=ERR %s --implicit-check-not=error:

# ERR-COUNT-3: {{.*}}.script:1: symbol not found: symbol

# MAP:      VMA              LMA     Size Align Out     In      Symbol
# MAP-NEXT:   0                0        0     1 symbol2 = symbol
# MAP-NEXT:   0                0        1     4 .text
# MAP-NEXT:   0                0        1     4         {{.*}}.o:(.text)
# MAP-NEXT:   0                0        0     1                 _start
# MAP-NEXT:   0                0        8     1 .comment
# MAP-NEXT:   0                0        8     1         <internal>:(.comment)
# MAP-NEXT:   0                0       60     8 .symtab
# MAP-NEXT:   0                0       60     8         <internal>:(.symtab)
# MAP-NEXT:   0                0       2a     1 .shstrtab
# MAP-NEXT:   0                0       2a     1         <internal>:(.shstrtab)
# MAP-NEXT:   0                0       17     1 .strtab
# MAP-NEXT:   0                0       17     1         <internal>:(.strtab)

.global _start
_start:
 nop
