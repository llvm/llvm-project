# REQUIRES: x86

# RUN: split-file %s %t --no-leading-lines

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -o %t/test -lSystem -lc++ -framework CoreFoundation %t/libNested.tbd %t/libTlvWeak.tbd %t/test.o
# RUN: llvm-objdump --bind --weak-bind --no-show-raw-insn -d -r %t/test | FileCheck %s

## Targeting an arch not listed in the tbd should fallback to an ABI compatible arch
# RUN: %lld -arch x86_64h -o %t/test-compat -lSystem -lc++ -framework CoreFoundation %t/libNested.tbd %t/libTlvWeak.tbd %t/test.o
# RUN: llvm-objdump --bind --weak-bind --no-show-raw-insn -d -r %t/test-compat | FileCheck %s

## Setting LD_DYLIB_CPU_SUBTYPES_MUST_MATCH forces exact target arch match.
# RUN: env LD_DYLIB_CPU_SUBTYPES_MUST_MATCH=1 not %lld -arch x86_64h -o /dev/null -lSystem -lc++ -framework \
# RUN:   CoreFoundation %t/libNested.tbd %t/libTlvWeak.tbd %t/test.o 2>&1 | FileCheck %s -check-prefix=INCOMPATIBLE

# INCOMPATIBLE:      error: {{.*}}libSystem.tbd(/usr/lib/libSystem.dylib) is incompatible with x86_64h (macOS)
# INCOMPATIBLE-NEXT: error: {{.*}}libc++.tbd(/usr/lib/libc++.dylib) is incompatible with x86_64h (macOS)
# INCOMPATIBLE-NEXT: error: {{.*}}CoreFoundation.tbd(/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation) is incompatible with x86_64h (macOS)

## libReexportSystem.tbd tests that we can reference symbols from a 2nd-level
## tapi document, re-exported by a top-level tapi document, which itself is
## re-exported by another top-level tapi document.
# RUN: %lld -o %t/with-reexport -lSystem -L%t %t/libReexportNested.tbd %t/libTlvWeak.tbd -lc++ -framework CoreFoundation %t/test.o
# RUN: llvm-objdump --bind --weak-bind --no-show-raw-insn -d -r %t/with-reexport | FileCheck %s

# CHECK: Bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_CLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_METACLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_IVAR_$_NSConstantArray._count
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_EHTYPE_$_NSException
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libc++abi      ___gxx_personality_v0
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libNested3     _deeply_nested
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libTlvWeak     _weak
# CHECK-DAG: __DATA __thread_ptrs {{.*}} pointer 0 libTlvWeak _tlv

# CHECK: Weak bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 _weak

# RUN: llvm-otool -l %t/test | FileCheck --check-prefix=LOAD %s

# RUN: llvm-otool -l %t/with-reexport | \
# RUN:     FileCheck --check-prefixes=LOAD,LOAD-REEXPORT %s

# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT:               cmdsize
# LOAD-NEXT:                  name /usr/lib/libSystem.dylib
# LOAD-NEXT:            time stamp
# LOAD-NEXT:       current version 1.1.1
# LOAD-NEXT: compatibility version

# LOAD-REEXPORT:          cmd LC_LOAD_DYLIB
# LOAD-REEXPORT-NEXT:               cmdsize
# LOAD-REEXPORT-NEXT:                  name /usr/lib/libReexportNested.dylib
# LOAD-REEXPORT-NEXT:            time stamp
# LOAD-REEXPORT-NEXT:       current version 1.0.0
# LOAD-REEXPORT-NEXT: compatibility version

#--- test.s
.section __TEXT,__text
.global _main

_main:
  mov _tlv@TLVP(%rip), %rax
  ret

.data
  .quad _OBJC_CLASS_$_NSObject
  .quad _OBJC_METACLASS_$_NSObject
  .quad _OBJC_IVAR_$_NSConstantArray._count
  .quad _OBJC_EHTYPE_$_NSException
## This symbol is defined in an inner TAPI document within libNested.tbd.
  .quad _deeply_nested

## This symbol is defined in libc++abi.tbd, but we are linking test.o against
## libc++.tbd (which re-exports libc++abi). Linking against this symbol verifies
## that .tbd file re-exports can refer not just to TAPI documents within the
## same .tbd file, but to other on-disk files as well.
  .quad ___gxx_personality_v0

  .quad _weak

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

#--- libReexportNested.tbd
--- !tapi-tbd-v3
archs:            [ i386, x86_64 ]
uuids:            [ 'i386: 00000000-0000-0000-0000-000000000000', 'x86_64: 00000000-0000-0000-0000-000000000001' ]
platform:         macosx
install-name:     '/usr/lib/libReexportNested.dylib'
exports:
  - archs:      [ i386, x86_64 ]
    re-exports: [ 'libNested.dylib' ]
...

#--- libTlvWeak.tbd
--- !tapi-tbd
tbd-version:      4
targets:          [ x86_64-macos ]
uuids:
  - target:       x86_64-macos
    value:        00000000-0000-0000-0000-000000000000
install-name:     '/usr/lib/libTlvWeak.dylib'
current-version:  0001.001.1
exports:                            # Validate weak & thread-local symbols 
  - targets:      [ x86_64-macos ]
    weak-symbols: [ _weak ]
    thread-local-symbols: [ _tlv ]
...
