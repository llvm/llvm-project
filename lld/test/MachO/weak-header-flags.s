# REQUIRES: x86
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libweak-defines.s -o %t/libweak-defines.o
# RUN: %lld -dylib %t/libweak-defines.o -o %t/libweak-defines.dylib
# RUN: llvm-readobj --file-headers %t/libweak-defines.dylib | FileCheck %s --check-prefix=WEAK-DEFINES-AND-BINDS

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/binds-to-weak.s -o %t/binds-to-weak.o
# RUN: %lld -lSystem -L%t -lweak-defines -o %t/binds-to-weak %t/binds-to-weak.o
# RUN: llvm-readobj --file-headers %t/binds-to-weak | FileCheck %s --check-prefix=WEAK-BINDS-ONLY

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/overrides-weak.s -o %t/overrides-weak.o
# RUN: %lld -lSystem -L%t -lweak-defines -o %t/overrides-weak %t/overrides-weak.o
# RUN: llvm-readobj --file-headers %t/overrides-weak | FileCheck %s --check-prefix=WEAK-DEFINES-ONLY

# WEAK-DEFINES-AND-BINDS: MH_BINDS_TO_WEAK
# WEAK-DEFINES-AND-BINDS: MH_WEAK_DEFINES

# WEAK-BINDS-ONLY-NOT:    MH_WEAK_DEFINES
# WEAK-BINDS-ONLY:        MH_BINDS_TO_WEAK
# WEAK-BINDS-ONLY-NOT:    MH_WEAK_DEFINES

# WEAK-DEFINES-ONLY-NOT:  MH_BINDS_TO_WEAK
# WEAK-DEFINES-ONLY:      MH_WEAK_DEFINES
# WEAK-DEFINES-ONLY-NOT:  MH_BINDS_TO_WEAK

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

#--- overrides-weak.s

.globl _main, _foo
_foo:

_main:
  ret
