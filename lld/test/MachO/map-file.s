# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c-string-literal.s -o %t/c-string-literal.o

# RUN: %lld -map %t/map %t/test.o %t/foo.o --time-trace -o %t/test-map
# RUN: llvm-objdump --syms --section-headers %t/test-map > %t/objdump
# RUN: cat %t/objdump %t/map > %t/out
# RUN: FileCheck %s < %t/out
# RUN: FileCheck %s --check-prefix=MAPFILE < %t/test-map.time-trace

# CHECK:      Sections:
# CHECK-NEXT: Idx  Name          Size           VMA           Type
# CHECK-NEXT: 0    __text        {{[0-9a-f]+}}  [[#%x,TEXT:]] TEXT
# CHECK-NEXT: 1    obj           {{[0-9a-f]+}}  [[#%x,DATA:]] DATA
# CHECK-NEXT: 2    __common      {{[0-9a-f]+}}  [[#%x,BSS:]]  BSS

# CHECK: SYMBOL TABLE:
# CHECK-NEXT: [[#%x,MAIN:]]   g     F __TEXT,__text _main
# CHECK-NEXT: [[#%x,NUMBER:]] g     O __DATA,__common _number
# CHECK-NEXT: [[#%x,FOO:]]    g     O __TEXT,obj _foo
# CHECK-NEXT: [[#%x,HEADER:]] g     F __TEXT,__text __mh_execute_header

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
# CHECK-NEXT: 0x[[#MAIN]]      [  1]  _main
# CHECK-NEXT: 0x[[#FOO]]       [  2]  _foo
# CHECK-NEXT: 0x[[#NUMBER]]    [  1]  _number

# RUN: %lld -map %t/c-string-literal-map %t/c-string-literal.o -o %t/c-string-literal-out
# RUN: FileCheck --check-prefix=CSTRING %s < %t/c-string-literal-map

## C-string literals should be printed as "literal string: <C string literal>"
# CSTRING-LABEL: Symbols:
# CSTRING-DAG: _main
# CSTRING-DAG: literal string: Hello world!\n
# CSTRING-DAG: literal string: Hello, it's me

# RUN: %lld -dead_strip -map %t/dead-c-string-literal-map %t/c-string-literal.o -o %t/dead-c-string-literal-out
# RUN: FileCheck --check-prefix=DEADCSTRING %s < %t/dead-c-string-literal-map

## C-string literals should be printed as "literal string: <C string literal>"
# DEADCSTRING-LABEL: Symbols:
# DEADCSTRING-DAG: _main
# DEADCSTRING-DAG: literal string: Hello world!\n
# DEADCSTRING-LABEL: Dead Stripped Symbols:
# DEADCSTRING-DAG: literal string: Hello, it's me

# MAPFILE: "name":"Total Write map file"

#--- foo.s
.section __TEXT,obj
.globl _foo
_foo:

#--- test.s
.comm _number, 1
.globl _main
_main:
  ret

#--- c-string-literal.s
.section __TEXT,__cstring
.globl _hello_world, _hello_its_me, _main

_hello_world:
.asciz "Hello world!\n"

_hello_its_me:
.asciz "Hello, it's me"

.text
_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _hello_world(%rip), %rsi
  mov $13, %rdx # length of str
  syscall
  ret
