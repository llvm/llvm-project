# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -map %t/map %t/test.o %t/foo.o -o %t/test-map
# RUN: llvm-objdump --syms --section-headers %t/test-map > %t/objdump
# RUN: cat %t/objdump %t/map > %t/out
# RUN: FileCheck %s < %t/out

#--- foo.s
.section __TEXT,obj
.globl _foo
_foo:

#--- test.s
.comm _number, 1
.globl _main
_main:
  ret

# CHECK:      Sections:
# CHECK-NEXT: Idx  Name          Size           VMA           Type
# CHECK-NEXT: 0    __text        {{[0-9a-f]+}}  [[#%x,TEXT:]] TEXT
# CHECK-NEXT: 1    obj           {{[0-9a-f]+}}  [[#%x,DATA:]] DATA
# CHECK-NEXT: 2    __common      {{[0-9a-f]+}}  [[#%x,BSS:]]  BSS

# CHECK: SYMBOL TABLE:
# CHECK-NEXT: [[#%x,MAIN:]]   g     F __TEXT,__text _main
# CHECK-NEXT: [[#%x,NUMBER:]] g     O __DATA,__common _number
# CHECK-NEXT: [[#%x,FOO:]]    g     O __TEXT,obj _foo
# CHECK-NEXT: {{0+}}          g       *ABS* __mh_execute_header

# CHECK-NEXT: # Path: {{.*}}{{/|\\}}map-file.s.tmp/test-map
# CHECK-NEXT: # Arch: x86_64
# CHECK-NEXT: # Object files:
# CHECK-NEXT: [  0] linker synthesized
# CHECK-NEXT: [  1] {{.*}}{{/|\\}}map-file.s.tmp/test.o
# CHECK-NEXT: [  2] {{.*}}{{/|\\}}map-file.s.tmp/foo.o

# CHECK-NEXT: # Sections:
# CHECK-NEXT: # Address    Size              Segment    Section
# CHECK-NEXT: 0x[[#TEXT]]  0x{{[0-9a-f]+}}   __TEXT  __text
# CHECK-NEXT: 0x[[#DATA]]  0x{{[0-9a-f]+}}   __TEXT  obj
# CHECK-NEXT: 0x[[#BSS]]   0x{{[0-9a-f]+}}   __DATA  __common

# CHECK-NEXT: # Symbols:
# CHECK-NEXT: # Address        File  Name
# CHECK-NEXT: 0x[[#NUMBER]]    [  1]  _number
# CHECK-NEXT: 0x[[#MAIN]]      [  1]  _main
# CHECK-NEXT: 0x[[#FOO]]       [  2]  _foo
