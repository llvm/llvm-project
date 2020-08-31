# REQUIRES: x86
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libweak-defines.s -o %t/libweak-defines.o
# RUN: lld -flavor darwinnew -syslibroot %S/Inputs/MacOSX.sdk -dylib %t/libweak-defines.o -o %t/libweak-defines.dylib
# RUN: llvm-readobj --file-headers %t/libweak-defines.dylib | FileCheck %s --check-prefix=WEAK-DEFINES-AND-BINDS

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/binds-to-weak.s -o %t/binds-to-weak.o
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -L%t -lweak-defines -o %t/binds-to-weak %t/binds-to-weak.o
# RUN: llvm-readobj --file-headers %t/binds-to-weak | FileCheck %s --check-prefix=WEAK-BINDS-ONLY

# WEAK-DEFINES-AND-BINDS: MH_BINDS_TO_WEAK
# WEAK-DEFINES-AND-BINDS: MH_WEAK_DEFINES

# WEAK-BINDS-ONLY-NOT:    MH_WEAK_DEFINES
# WEAK-BINDS-ONLY:        MH_BINDS_TO_WEAK
# WEAK-BINDS-ONLY-NOT:    MH_WEAK_DEFINES

#--- libweak-defines.s

.globl _foo
.weak_definition _foo
_foo:
  ret

#--- binds-to-weak.s

.globl _main
_main:
  callq _foo
  ret

## Don't generate MH_WEAK_DEFINES for weak locals
.weak_definition _weak_local
_weak_local:
