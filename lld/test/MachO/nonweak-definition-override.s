# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/nonweakdef.s -o %t/nonweakdef.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakdef.s -o %t/weakdef.o
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk -dylib %t/libfoo.o -o %t/libfoo.dylib

## Check that non-weak defined symbols override weak dylib symbols.
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk %t/nonweakdef.o -L%t -lfoo -o %t/nonweakdef -lSystem
# RUN: llvm-objdump --macho --weak-bind %t/nonweakdef | FileCheck %s

## Test loading the dylib before the obj file.
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk -L%t -lfoo %t/nonweakdef.o -o %t/nonweakdef -lSystem
# RUN: llvm-objdump --macho --weak-bind %t/nonweakdef | FileCheck %s

# CHECK:       Weak bind table:
# CHECK-NEXT:  segment  section            address     type       addend   symbol
# CHECK-NEXT:                                          strong              _weak_in_dylib
# CHECK-EMPTY:

## Check that weak defined symbols do not override weak dylib symbols.
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk %t/weakdef.o -L%t -lfoo -o %t/weakdef -lSystem
# RUN: llvm-objdump --macho --weak-bind %t/weakdef | FileCheck %s --check-prefix=NO-WEAK-OVERRIDE

## Test loading the dylib before the obj file.
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk -L%t -lfoo %t/weakdef.o -o %t/weakdef -lSystem
# RUN: llvm-objdump --macho --weak-bind %t/weakdef | FileCheck %s --check-prefix=NO-WEAK-OVERRIDE

# NO-WEAK-OVERRIDE:       Weak bind table:
# NO-WEAK-OVERRIDE-NEXT:  segment section address type addend symbol
# NO-WEAK-OVERRIDE-EMPTY:

#--- libfoo.s

.globl _weak_in_dylib, _nonweak_in_dylib
.weak_definition _weak_in_dylib

_weak_in_dylib:
_nonweak_in_dylib:

#--- nonweakdef.s

.globl _main, _weak_in_dylib, _nonweak_in_dylib

_weak_in_dylib:
_nonweak_in_dylib:

_main:
  ret

#--- weakdef.s

.globl _main, _weak_in_dylib, _nonweak_in_dylib
.weak_definition _weak_in_dylib, _nonweak_in_dylib

_weak_in_dylib:
_nonweak_in_dylib:

_main:
  ret
