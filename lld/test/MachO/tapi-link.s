# REQUIRES: x86

# RUN: split-file %s %t --no-leading-lines

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -o %t/test -lSystem -lc++ -framework CoreFoundation %t/libNested.tbd %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d -r %t/test | FileCheck %s

## libReexportSystem.tbd tests that we can reference symbols from a dylib,
## re-exported by a top-level tapi document, which itself is re-exported by
## another top-level tapi document.
# RUN: %lld -o %t/with-reexport %S/Inputs/libReexportSystem.tbd -lc++ -framework CoreFoundation %t/libNested.tbd %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d -r %t/with-reexport | FileCheck %s

# CHECK: Disassembly of section __TEXT,__text:
# CHECK: movq {{.*}} # [[ADDR:[0-9a-f]+]]

# CHECK: Bind table:
# CHECK-DAG: __DATA_CONST __got 0x[[ADDR]] pointer 0 libSystem ___nan
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_CLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_METACLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_IVAR_$_NSConstantArray._count
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_EHTYPE_$_NSException
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libc++abi      ___gxx_personality_v0
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libNested3     _deeply_nested

# RUN: llvm-otool -l %t/test | FileCheck --check-prefix=LOAD %s

# RUN: llvm-otool -l %t/with-reexport | \
# RUN:     FileCheck --check-prefixes=LOAD-REEXPORT,LOAD %s

# LOAD-REEXPORT:          cmd LC_LOAD_DYLIB
# LOAD-REEXPORT-NEXT:               cmdsize
# LOAD-REEXPORT-NEXT:                  name /usr/lib/libReexportSystem.dylib
# LOAD-REEXPORT-NEXT:            time stamp
# LOAD-REEXPORT-NEXT:       current version 1.0.0
# LOAD-REEXPORT-NEXT: compatibility version

# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT:               cmdsize
# LOAD-NEXT:                  name /usr/lib/libSystem.dylib
# LOAD-NEXT:            time stamp
# LOAD-NEXT:       current version 1.1.1
# LOAD-NEXT: compatibility version

#--- test.s
.section __TEXT,__text
.global _main

_main:
## This symbol is defined in an inner TAPI document within libSystem.tbd.
  movq ___nan@GOTPCREL(%rip), %rax
  ret

.data
  .quad _OBJC_CLASS_$_NSObject
  .quad _OBJC_METACLASS_$_NSObject
  .quad _OBJC_IVAR_$_NSConstantArray._count
  .quad _OBJC_EHTYPE_$_NSException
  .quad _deeply_nested

## This symbol is defined in libc++abi.tbd, but we are linking test.o against
## libc++.tbd (which re-exports libc++abi). Linking against this symbol verifies
## that .tbd file re-exports can refer not just to TAPI documents within the
## same .tbd file, but to other on-disk files as well.
  .quad ___gxx_personality_v0

## This tests that we can locate a symbol re-exported by a child of a TAPI
## document.
#--- libNested.tbd
--- !tapi-tbd-v3
archs:            [ x86_64 ]
uuids:            [ 'x86_64: 00000000-0000-0000-0000-000000000000' ]
platform:         macosx
install-name:     '/usr/lib/libNested.dylib'
exports:
  - archs:      [ x86_64 ]
    re-exports: [ '/usr/lib/libNested2.dylib' ]
--- !tapi-tbd-v3
archs:            [ x86_64 ]
uuids:            [ 'x86_64: 00000000-0000-0000-0000-000000000001' ]
platform:         macosx
install-name:     '/usr/lib/libNested2.dylib'
exports:
  - archs:      [ x86_64 ]
    re-exports: [ '/usr/lib/libNested3.dylib' ]
--- !tapi-tbd-v3
archs:            [ x86_64 ]
uuids:            [ 'x86_64: 00000000-0000-0000-0000-000000000002' ]
platform:         macosx
install-name:     '/usr/lib/libNested3.dylib'
exports:
  - archs:      [ x86_64 ]
    symbols:    [ _deeply_nested ]
...
