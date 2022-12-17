# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c-string-literal.s -o %t/c-string-literal.o

# RUN: %lld -demangle -map %t/map %t/test.o %t/foo.o %t/c-string-literal.o \
# RUN:   --time-trace -o %t/test
# RUN: llvm-objdump --syms --section-headers %t/test > %t/objdump
## Check that symbols in cstring sections aren't emitted
## Also check that we don't have redundant EH_Frame symbols (regression test)
# RUN: cat %t/objdump %t/map | FileCheck %s --implicit-check-not _hello_world \
# RUN:   --implicit-check-not EH_Frame
# RUN: FileCheck %s --check-prefix=MAPFILE < %t/test.time-trace

# CHECK:      Sections:
# CHECK-NEXT: Idx  Name          Size           VMA               Type
# CHECK-NEXT: 0    __text        {{[0-9a-f]+}}  [[#%x,TEXT:]]     TEXT
# CHECK-NEXT: 1    __cstring     {{[0-9a-f]+}}  [[#%x,CSTR:]]     DATA
# CHECK-NEXT: 2    __unwind_info {{[0-9a-f]+}}  [[#%x,UNWIND:]]   DATA
# CHECK-NEXT: 3    __eh_frame    {{[0-9a-f]+}}  [[#%x,EH_FRAME:]] DATA
# CHECK-NEXT: 4    __common      {{[0-9a-f]+}}  [[#%x,BSS:]]      BSS

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,MAIN:]]    g     F __TEXT,__text _main
# CHECK-DAG:  [[#%x,NUMBER:]]  g     O __DATA,__common _number
# CHECK-DAG:  [[#%x,BAR:]]     g     F __TEXT,__text _bar
# CHECK-DAG:  [[#%x,FOO:]]     g     F __TEXT,__text __ZTIN3foo3bar4MethE
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
# CHECK-NEXT: # Address           Size              Segment  Section
# CHECK-NEXT: 0x[[#%X,TEXT]]      0x{{[0-9A-F]+}}   __TEXT   __text
# CHECK-NEXT: 0x[[#%X,CSTR]]      0x{{[0-9A-F]+}}   __TEXT   __cstring
# CHECK-NEXT: 0x[[#%X,UNWIND]]    0x{{[0-9A-F]+}}   __TEXT   __unwind_info
# CHECK-NEXT: 0x[[#%X,EH_FRAME]]  0x{{[0-9A-F]+}}   __TEXT   __eh_frame
# CHECK-NEXT: 0x[[#%X,BSS]]       0x{{[0-9A-F]+}}   __DATA   __common

# CHECK-NEXT: # Symbols:
# CHECK-NEXT: # Address                  Size        File   Name
# CHECK-DAG:  0x[[#%X,MAIN]]             0x00000001  [  1]  _main
# CHECK-DAG:  0x[[#%X,BAR]]              0x00000001  [  1]  _bar
# CHECK-DAG:  0x[[#%X,FOO]]              0x00000001  [  2]  __ZTIN3foo3bar4MethE
# CHECK-DAG:  0x[[#%X,HIWORLD]]          0x0000000E  [  3]  literal string: Hello world!\n
# CHECK-DAG:  0x[[#%X,HIITSME]]          0x0000000F  [  3]  literal string: Hello, it's me
# CHECK-DAG:  0x[[#%X,HIITSME + 0xf]]    0x0000000E  [  3]  literal string: Hello world!\n
# CHECK-DAG:  0x[[#%X,NUMBER]]           0x00000001  [  1]  _number
# CHECK-DAG:  0x[[#%X,UNWIND]]           0x0000103C  [  0]  compact unwind info
## Note: ld64 prints "CIE" and "FDE for: <function>" instead of "EH_Frame".
# CHECK-DAG:  0x[[#%X,EH_FRAME]]         0x00000018  [  1]  EH_Frame
# CHECK-DAG:  0x[[#%X,EH_FRAME + 0x18]]  0x00000020  [  1]  EH_Frame

# MAPFILE: "name":"Total Write map file"

# RUN: %lld -demangle -dead_strip -map %t/stripped-map %t/test.o %t/foo.o %t/c-string-literal.o -o %t/stripped
# RUN: FileCheck --check-prefix=STRIPPED %s < %t/stripped-map

## C-string literals should be printed as "literal string: <C string literal>"
# STRIPPED-LABEL: Dead Stripped Symbols:
# STRIPPED-DAG:   <<dead>>	0x00000001	[  1] _bar
# STRIPPED-DAG:   <<dead>>	0x00000001	[  1] _number
# STRIPPED-DAG:   <<dead>>	0x00000001	[  2] __ZTIN3foo3bar4MethE
# STRIPPED-DAG:   <<dead>>	0x0000000E	[  3] literal string: Hello world!\n
# STRIPPED-DAG:   <<dead>>	0x0000000F	[  3] literal string: Hello, it's me
# STRIPPED-DAG:   <<dead>>	0x0000000E	[  3] literal string: Hello world!\n

# RUN: %lld --icf=all -map %t/icf-map %t/test.o %t/foo.o %t/c-string-literal.o -o %t/icf
# RUN: FileCheck --check-prefix=ICF %s < %t/icf-map

## Verify that folded symbols and cstrings have size zero. Note that ld64 prints
## folded symbols but not folded cstrings; we print both.

# ICF:     Symbols:
# ICF-DAG: 0x[[#%X,FOO:]]     0x00000000  [  2] __ZTIN3foo3bar4MethE
# ICF-DAG: 0x[[#FOO]]         0x00000001  [  1] _bar
# ICF-DAG: 0x[[#%X,HIWORLD:]] 0x0000000E  [  3]  literal string: Hello world!\n
# ICF-DAG: 0x[[#%X,HIWORLD]]  0x00000000  [  3]  literal string: Hello world!\n

#--- foo.s
.globl __ZTIN3foo3bar4MethE
## This C++ symbol makes it clear that we do not print the demangled name in
## the map file, even if `-demangle` is passed.
__ZTIN3foo3bar4MethE:
  nop

.subsections_via_symbols

#--- test.s
.comm _number, 1
.globl _main, _bar

_main:
.cfi_startproc
.cfi_def_cfa_offset 16
  ret
.cfi_endproc

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

.asciz "Hello world!\n"

.subsections_via_symbols
