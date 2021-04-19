# REQUIRES: x86
## Test -r --gc-sections.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## By default all regular sections are discarded. We currently don't track
## usage of group signature symbols and will retain them and their associated
## STT_SECTION symbols.
# RUN: ld.lld -r --gc-sections --print-gc-sections %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck %s

# CHECK:      [ 1] .group
# CHECK-NEXT: [ 2] .note.GNU-stack

# CHECK:      Symbol table '.symtab' contains 3 entries:
# CHECK-NEXT: Num:
# CHECK-NEXT: 0:
# CHECK-NEXT: 1: {{.*}} NOTYPE  LOCAL DEFAULT  1 group
# CHECK-NEXT: 2: {{.*}} SECTION LOCAL DEFAULT  1

## -u keeps .text.bar alive. Other group members are kept alive as well.
# RUN: ld.lld -r --gc-sections -u bar %t.o -o - | llvm-readelf -Ss - | \
# RUN:   FileCheck %s --check-prefix=KEEP_GROUP
## -e, --init and --fini are similar.
# RUN: ld.lld -r --gc-sections -e bar %t.o -o - | llvm-readelf -Ss - | \
# RUN:   FileCheck %s --check-prefix=KEEP_GROUP
# RUN: ld.lld -r --gc-sections --init=bar %t.o -o - | llvm-readelf -Ss - | \
# RUN:   FileCheck %s --check-prefix=KEEP_GROUP
# RUN: ld.lld -r --gc-sections --fini=bar %t.o -o - | llvm-readelf -Ss - | \
# RUN:   FileCheck %s --check-prefix=KEEP_GROUP

# KEEP_GROUP:      [ 1] .group
# KEEP_GROUP-NEXT: [ 2] .text.bar
# KEEP_GROUP-NEXT: [ 3] .text.foo
# KEEP_GROUP-NEXT: [ 4] .note.GNU-stack

# KEEP_GROUP:      Symbol table '.symtab' contains 7 entries:
# KEEP_GROUP:      4: {{.*}} SECTION
# KEEP_GROUP-NEXT: 5: {{.*}}   2 bar
# KEEP_GROUP-NEXT: 6: {{.*}}   3 foo

## If .text is retained, its referenced qux and .fred are retained as well.
## fred_und is used (by .fred) and thus emitted.
## Note, GNU ld does not retain qux.
# RUN: ld.lld -r --gc-sections -z nostart-stop-gc -e _start %t.o -o %tstart.ro
# RUN: llvm-readelf -Ss %tstart.ro | FileCheck %s --check-prefix=KEEP_START

# KEEP_START:      [ 1] .text
# KEEP_START-NEXT: [ 2] .rela.text
# KEEP_START-NEXT: [ 3] qux
# KEEP_START-NEXT: [ 4] .group
# KEEP_START-NEXT: [ 5] .fred
# KEEP_START-NEXT: [ 6] .rela.fred
# KEEP_START-NEXT: [ 7] .note.GNU-stack

# KEEP_START:      Symbol table '.symtab' contains 10 entries:
# KEEP_START:      5: {{.*}} SECTION
# KEEP_START-NEXT: 6: {{.*}}   1 _start
# KEEP_START-NEXT: 7: {{.*}}   5 fred
# KEEP_START-NEXT: 8: {{.*}} UND __start_qux
# KEEP_START-NEXT: 9: {{.*}} UND fred_und

.section qux,"a",@progbits
  .byte 0

.text
.globl _start, bar, foo, fred
_start:
  call fred
  .quad __start_qux

.section .text.bar,"axG",@progbits,group,comdat
bar:
  .byte 1
.section .text.foo,"axG",@progbits,group,comdat
foo:
  .byte 2
.section .fred,"ax",@progbits
fred:
  call fred_und
