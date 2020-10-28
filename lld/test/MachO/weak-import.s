# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: %lld -lSystem -dylib %t/foo.o -o %t/libfoo.dylib

# RUN: %lld -weak-lSystem %t/test.o -weak_framework CoreFoundation -weak_library %t/libfoo.dylib -o %t/test
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s -DDIR=%t

# CHECK:          cmd LC_LOAD_WEAK_DYLIB
# CHECK-NEXT: cmdsize
# CHECK-NEXT:    name /usr/lib/libSystem.B.dylib

# CHECK:          cmd LC_LOAD_WEAK_DYLIB
# CHECK-NEXT: cmdsize
# CHECK-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation

# CHECK:          cmd LC_LOAD_WEAK_DYLIB
# CHECK-NEXT: cmdsize
# CHECK-NEXT:    name [[DIR]]/libfoo.dylib

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.globl _main
.text
_main:
  ret
