# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c-string-literal.s -o %t/c-string-literal.o

# RUN: %lld -map %t/map %t/test.o %t/foo.o %t/c-string-literal.o --time-trace -o %t/test
# RUN: llvm-objdump --syms --section-headers %t/test > %t/objdump
# RUN: cat %t/objdump %t/map > %t/out
# RUN: FileCheck %s < %t/out
# RUN: FileCheck %s --check-prefix=MAPFILE < %t/test.time-trace

# CHECK:      Sections:
# CHECK-NEXT: Idx  Name          Size           VMA           Type
# CHECK-NEXT: 0    __text        {{[0-9a-f]+}}  [[#%x,TEXT:]] TEXT
# CHECK-NEXT: 1    obj           {{[0-9a-f]+}}  [[#%x,DATA:]] TEXT
# CHECK-NEXT: 2    __cstring     {{[0-9a-f]+}}  [[#%x,CSTR:]] DATA
# CHECK-NEXT: 3    __common      {{[0-9a-f]+}}  [[#%x,BSS:]]  BSS

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,MAIN:]]    g     F __TEXT,__text _main
# CHECK-DAG:  [[#%x,NUMBER:]]  g     O __DATA,__common _number
# CHECK-DAG:  [[#%x,FOO:]]     g     F __TEXT,obj _foo
# CHECK-DAG:  [[#%x,HIWORLD:]] g     O __TEXT,__cstring _hello_world
# CHECK-DAG:  [[#%x,HIITSME:]] g     O __TEXT,__cstring _hello_its_me

# CHECK:      # Path: {{.*}}{{/|\\}}map-file.s.tmp/test
# CHECK-NEXT: # Arch: x86_64
# CHECK-NEXT: # Object files:
# CHECK-NEXT: [  0] linker synthesized
# CHECK-NEXT: [  1] {{.*}}{{/|\\}}map-file.s.tmp/test.o
# CHECK-NEXT: [  2] {{.*}}{{/|\\}}map-file.s.tmp/foo.o
# CHECK-NEXT: [  3] {{.*}}{{/|\\}}map-file.s.tmp/c-string-literal.o

# CHECK-NEXT: # Sections:
# CHECK-NEXT: # Address       Size              Segment  Section
# CHECK-NEXT: 0x[[#%X,TEXT]]  0x{{[0-9A-F]+}}   __TEXT   __text
# CHECK-NEXT: 0x[[#%X,DATA]]  0x{{[0-9A-F]+}}   __TEXT   obj
# CHECK-NEXT: 0x[[#%X,CSTR]]  0x{{[0-9A-F]+}}   __TEXT   __cstring
# CHECK-NEXT: 0x[[#%X,BSS]]   0x{{[0-9A-F]+}}   __DATA   __common

# CHECK-NEXT: # Symbols:
# CHECK-NEXT: # Address           Size        File   Name
# CHECK-DAG:  0x[[#%X,MAIN]]      0x00000001  [  1]  _main
# CHECK-DAG:  0x[[#%X,FOO]]       0x00000001  [  2]  _foo
# CHECK-DAG:  0x[[#%X,HIWORLD]]   0x0000000E  [  3]  literal string: Hello world!\n
# CHECK-DAG:  0x[[#%X,HIITSME]]   0x0000000F  [  3]  literal string: Hello, it's me
# CHECK-DAG:  0x[[#%X,NUMBER]]    0x00000001  [  1]  _number

# MAPFILE: "name":"Total Write map file"

# RUN: %lld -dead_strip -map %t/stripped-map %t/test.o %t/foo.o %t/c-string-literal.o -o %t/stripped
# RUN: FileCheck --check-prefix=STRIPPED %s < %t/stripped-map

## C-string literals should be printed as "literal string: <C string literal>"
# STRIPPED-LABEL: Dead Stripped Symbols:
# STRIPPED-DAG:   <<dead>> 0x00000001 [  2] _foo
# STRIPPED-DAG:   <<dead>> 0x0000000E [  3] literal string: Hello world!\n
# STRIPPED-DAG:   <<dead>> 0x0000000F [  3] literal string: Hello, it's me
# STRIPPED-DAG:   <<dead>> 0x00000001 [  1] _number

# RUN: %lld --icf=all -map %t/icf-map %t/test.o %t/foo.o %t/c-string-literal.o -o %t/icf
# RUN: FileCheck --check-prefix=ICF %s < %t/icf-map

# ICF:     Symbols:
# ICF-DAG: 0x[[#%X,FOO:]]  0x00000000  [  2] _foo
# ICF-DAG: 0x[[#FOO]]      0x00000001  [  1] _bar

#--- foo.s
## ICF will only fold sections marked as pure_instructions
.section __TEXT,obj,regular,pure_instructions
.globl _foo
.alt_entry _alt_foo
_foo:
  nop

.subsections_via_symbols

#--- test.s
.comm _number, 1
.globl _main, _bar
.alt_entry _alt_bar

_main:
  ret

.section __TEXT,obj,regular,pure_instructions
_bar:
  nop

.subsections_via_symbols

#--- c-string-literal.s
.globl _hello_world, _hello_its_me

.cstring

_hello_world:
.asciz "Hello world!\n"

_hello_its_me:
.asciz "Hello, it's me"

.subsections_via_symbols
