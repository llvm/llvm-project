# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c-string-literal.s -o %t/c-string-literal.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/baz.s -o %t/baz.o

# RUN: llvm-ar --format=darwin crs %t/libfoo.a %t/foo.o
# RUN: %lld -dylib %t/baz.o -o %t/libbaz.dylib
# RUN: %lld -demangle -map %t/map %t/test.o -force_load %t/libfoo.a \
# RUN:   %t/c-string-literal.o %t/libbaz.dylib --time-trace -o %t/test \
# RUN:   --no-deduplicate-strings
# RUN: llvm-objdump --syms --section-headers %t/test > %t/objdump
## Check that symbols in cstring sections aren't emitted
## Also check that we don't have redundant EH_Frame symbols (regression test)
# RUN: cat %t/objdump %t/map | FileCheck %s --implicit-check-not _hello_world \
# RUN:   --implicit-check-not EH_Frame
# RUN: FileCheck %s --check-prefix=MAPFILE < %t/test.time-trace

# CHECK:       Sections:
# CHECK-NEXT:  Idx  Name         Size     VMA               Type
# CHECK-NEXT:  0 __text          0000001c [[#%x,TEXT:]]     TEXT
# CHECK-NEXT:  1 __stubs         0000000c [[#%x,STUBS:]]    TEXT
# CHECK-NEXT:  2 __stub_helper   0000001a [[#%x,HELPER:]]   TEXT
# CHECK-NEXT:  3 __cstring       0000002b [[#%x,CSTR:]]     DATA
# CHECK-NEXT:  4 __unwind_info   0000103c [[#%x,UNWIND:]]   DATA
# CHECK-NEXT:  5 __eh_frame      00000038 [[#%x,EH_FRAME:]] DATA
# CHECK-NEXT:  6 __got           00000010 [[#%x,GOT:]]      DATA
# CHECK-NEXT:  7 __la_symbol_ptr 00000010 [[#%x,LAZY:]]     DATA
# CHECK-NEXT:  8 __data          00000008 [[#%x,DATA:]]     DATA
# CHECK-NEXT:  9 __thread_ptrs   00000008 [[#%x,TLVP:]]     DATA
# CHECK-NEXT: 10 __common        00000001 [[#%x,BSS:]]      BSS

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,DYLD:]]    l     O __DATA,__data __dyld_private
# CHECK-DAG:  [[#%x,MAIN:]]    g     F __TEXT,__text _main
# CHECK-DAG:  [[#%x,NUMBER:]]  g     O __DATA,__common _number
# CHECK-DAG:  [[#%x,BAR:]]     w     F __TEXT,__text _bar
# CHECK-DAG:  [[#%x,FOO:]]     g     F __TEXT,__text __ZTIN3foo3bar4MethE
# CHECK-DAG:  [[#%x,HIWORLD:]] g     O __TEXT,__cstring _hello_world
# CHECK-DAG:  [[#%x,HIITSME:]] g     O __TEXT,__cstring _hello_its_me

# CHECK:      # Path: {{.*}}{{/|\\}}map-file.s.tmp/test
# CHECK-NEXT: # Arch: x86_64
# CHECK-NEXT: # Object files:
# CHECK-NEXT: [  0] linker synthesized
# CHECK-NEXT: [  1] {{.*}}{{/|\\}}usr/lib{{/|\\}}libSystem.tbd{{$}}
# CHECK-NEXT: [  2] {{.*}}{{/|\\}}map-file.s.tmp/test.o{{$}}
# CHECK-NEXT: [  3] {{.*}}{{/|\\}}map-file.s.tmp/libfoo.a(foo.o){{$}}
# CHECK-NEXT: [  4] {{.*}}{{/|\\}}map-file.s.tmp/c-string-literal.o{{$}}
# CHECK-NEXT: [  5] {{.*}}{{/|\\}}map-file.s.tmp/libbaz.dylib{{$}}

# CHECK-NEXT: # Sections:
# CHECK-NEXT: # Address           Size            Segment  Section
# CHECK-NEXT: 0x[[#%X,TEXT]]      0x{{[0-9A-F]+}} __TEXT  __text
# CHECK-NEXT: 0x[[#%X,STUBS]]     0x{{[0-9A-F]+}} __TEXT  __stubs
# CHECK-NEXT: 0x[[#%X,HELPER]]    0x{{[0-9A-F]+}} __TEXT  __stub_helper
# CHECK-NEXT: 0x[[#%X,CSTR]]      0x{{[0-9A-F]+}} __TEXT  __cstring
# CHECK-NEXT: 0x[[#%X,UNWIND]]    0x{{[0-9A-F]+}} __TEXT  __unwind_info
# CHECK-NEXT: 0x[[#%X,EH_FRAME]]  0x{{[0-9A-F]+}} __TEXT  __eh_frame
# CHECK-NEXT: 0x[[#%X,GOT]]       0x{{[0-9A-F]+}} __DATA_CONST  __got
# CHECK-NEXT: 0x[[#%X,LAZY]]      0x{{[0-9A-F]+}} __DATA  __la_symbol_ptr
# CHECK-NEXT: 0x[[#%X,DATA]]      0x{{[0-9A-F]+}} __DATA  __data
# CHECK-NEXT: 0x[[#%X,TLVP]]      0x{{[0-9A-F]+}} __DATA  __thread_ptrs
# CHECK-NEXT: 0x[[#%X,BSS]]       0x{{[0-9A-F]+}} __DATA  __common

# CHECK-NEXT: # Symbols:
# CHECK-NEXT: # Address                Size        File   Name
# CHECK-NEXT: 0x[[#%X,MAIN]]           0x00000019  [  2] _main
# CHECK-NEXT: 0x[[#%X,BAR]]            0x00000001  [  2] _bar
# CHECK-NEXT: 0x[[#%X,FOO]]            0x00000001  [  3] __ZTIN3foo3bar4MethE
# CHECK-NEXT: 0x[[#%X,FOO+1]]          0x00000001  [  3] ltmp1
# CHECK-NEXT: 0x[[#%X,STUBS]]          0x00000006  [  5] _baz
# CHECK-NEXT: 0x[[#%X,STUBS+6]]        0x00000006  [  2] _bar
# CHECK-NEXT: 0x[[#%X,HELPER]]         0x0000001A  [  0] helper helper
# CHECK-NEXT: 0x[[#%X,HIWORLD]]        0x0000000E  [  4] literal string: Hello world!\n
# CHECK-NEXT: 0x[[#%X,HIITSME]]        0x0000000F  [  4] literal string: Hello, it's me
# CHECK-NEXT: 0x[[#%X,HIITSME+0xf]]    0x0000000E  [  4] literal string: Hello world!\n
# CHECK-NEXT: 0x[[#%X,UNWIND]]         0x0000103C  [  0] compact unwind info
## Note: ld64 prints "CIE" and "FDE for: <function>" instead of "EH_Frame".
# CHECK-NEXT: 0x[[#%X,EH_FRAME]]       0x00000018  [  2] EH_Frame
# CHECK-NEXT: 0x[[#%X,EH_FRAME+0x18]]  0x00000020  [  2] EH_Frame
# CHECK-NEXT: 0x[[#%X,GOT]]            0x00000008  [  0] non-lazy-pointer-to-local: _baz2
# CHECK-NEXT: 0x[[#%X,GOT+8]]          0x00000008  [  0] non-lazy-pointer-to-local: dyld_stub_binder
# CHECK-NEXT: 0x[[#%X,LAZY]]           0x00000008  [  5] _baz
# CHECK-NEXT: 0x[[#%X,LAZY+8]]         0x00000008  [  2] _bar
# CHECK-NEXT: 0x[[#%X,DYLD]]           0x00000000  [  0] __dyld_private
# CHECK-NEXT: 0x[[#%X,TLVP]]           0x00000008  [  0] non-lazy-pointer-to-local: _baz_tlv
# CHECK-NEXT: 0x[[#%X,BSS]]            0x00000001  [  2] _number
# CHECK-EMPTY:

# MAPFILE: "name":"Total Write map file"

# RUN: %lld -demangle -dead_strip -map %t/stripped-map %t/test.o -force_load \
# RUN:   %t/libfoo.a %t/c-string-literal.o %t/libbaz.dylib -o %t/stripped
# RUN: FileCheck --check-prefix=STRIPPED %s < %t/stripped-map

# STRIPPED-LABEL: Dead Stripped Symbols:
# STRIPPED-DAG:   <<dead>>  0x00000001  [  2] _number
# STRIPPED-DAG:   <<dead>>  0x00000001  [  3] __ZTIN3foo3bar4MethE
# STRIPPED-DAG:   <<dead>>  0x0000000E  [  4] literal string: Hello world!\n
# STRIPPED-DAG:   <<dead>>  0x0000000F  [  4] literal string: Hello, it's me
# STRIPPED-DAG:   <<dead>>  0x0000000E  [  4] literal string: Hello world!\n

# RUN: %lld --icf=all -map %t/icf-map %t/test.o -force_load %t/libfoo.a \
# RUN:   %t/c-string-literal.o %t/libbaz.dylib -o /dev/null
# RUN: FileCheck --check-prefix=ICF %s < %t/icf-map

## Verify that folded symbols and cstrings have size zero. Note that ld64 prints
## folded symbols but not folded cstrings; we print both.

# ICF:     Symbols:
# ICF-DAG: 0x[[#%X,FOO:]]     0x00000000  [  3] __ZTIN3foo3bar4MethE
# ICF-DAG: 0x[[#FOO]]         0x00000001  [  2] _bar
# ICF-DAG: 0x[[#%X,HIWORLD:]] 0x0000000E  [  4]  literal string: Hello world!\n
# ICF-DAG: 0x[[#%X,HIWORLD]]  0x00000000  [  4]  literal string: Hello world!\n

#--- foo.s
.globl __ZTIN3foo3bar4MethE
## This should not appear in the map file since it is a zero-size private label
## symbol.
ltmp0:
## This C++ symbol makes it clear that we do not print the demangled name in
## the map file, even if `-demangle` is passed.
__ZTIN3foo3bar4MethE:
  nop

## This private label symbol will appear in the map file since it has nonzero
## size.
ltmp1:
  nop

.subsections_via_symbols

#--- test.s
.comm _number, 1
.globl _main, _bar
.weak_definition _bar

_main:
.cfi_startproc
.cfi_def_cfa_offset 16
  callq _bar
  callq _baz
  movq _baz2@GOTPCREL(%rip), %rax
  mov _baz_tlv@TLVP(%rip), %rax
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

#--- baz.s
.globl _baz, _baz2

_baz:
  nop

_baz2:
  nop

.section __DATA,__thread_vars,thread_local_variables
.globl _baz_tlv
_baz_tlv:
  nop
