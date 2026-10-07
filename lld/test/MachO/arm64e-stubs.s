# REQUIRES: aarch64
## On arm64e, calls to dylib functions go through __auth_stubs, which load the
## target from a signed __auth_got slot and authenticate it against the slot's
## address with braa, as ld64 does. Fast ObjC stubs reach _objc_msgSend the
## same way; small ones branch to its stub.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/libfoo.dylib -objc_stubs_fast -o %t/test
# RUN: llvm-objdump --macho --section-headers %t/test > %t/out
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn \
# RUN:   --section=__TEXT,__auth_stubs --section=__TEXT,__objc_stubs \
# RUN:   %t/test >> %t/out
# RUN: llvm-objdump --macho --dyld-info %t/test >> %t/out
# RUN: FileCheck %s < %t/out --implicit-check-not=__stubs \
# RUN:   --implicit-check-not=__got

# CHECK:      __text
# CHECK-NEXT: __auth_stubs {{0*}}00000010
# CHECK:      __auth_got {{[0-9a-f]+}} {{0*}}[[#%x,GOT:]] DATA

# CHECK-LABEL: _main:
# CHECK-NEXT:  bl {{.*}} ; symbol stub for: _foo
# CHECK-LABEL: Contents of (__TEXT,__auth_stubs) section
# CHECK-NEXT:  adrp x17, {{.*}} ; 0x[[#GOT]]
# CHECK-NEXT:  add x17, x17, #[[#%u,FOO:]] ; literal pool symbol address: _foo
# CHECK-NEXT:  ldr x16, [x17]
# CHECK-NEXT:  braa x16, x17
# CHECK-LABEL: Contents of (__TEXT,__objc_stubs) section
# CHECK-NEXT:  _objc_msgSend$foo:
# CHECK-NEXT:  adrp x1
# CHECK-NEXT:  ldr x1, [x1
# CHECK-NEXT:  adrp x17, {{.*}} ; 0x[[#GOT]]
# CHECK-NEXT:  add x17, x17, #[[#%u,MSGSEND:]]
# CHECK-NEXT:  ldr x16, [x17]
# CHECK-NEXT:  braa x16, x17
# CHECK-NEXT:  brk #0x1
# CHECK-NEXT:  brk #0x1

## Both __auth_got slots are authenticated binds, of _objc_msgSend and _foo,
## with the IA key, address diversity and no extra discriminator.
# CHECK-LABEL: dyld information:
# CHECK:       __DATA_CONST __auth_got 0x[[#%X,GOT]] 0xC009000000000000 bind 0x0000 true IA 0x0 libfoo _objc_msgSend
# CHECK-NEXT:  __DATA_CONST __auth_got 0x[[#%X,GOT+8]] 0xC001000000000001 bind 0x0000 true IA 0x0 libfoo _foo

## Small ObjC stubs keep branching to _objc_msgSend's stub.
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/libfoo.dylib -objc_stubs_small -o %t/small
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn \
# RUN:   --section=__TEXT,__objc_stubs %t/small | FileCheck %s --check-prefix=SMALL

# SMALL-LABEL: Contents of (__TEXT,__objc_stubs) section
# SMALL-NEXT:  _objc_msgSend$foo:
# SMALL-NEXT:  adrp x1
# SMALL-NEXT:  ldr x1, [x1
# SMALL-NEXT:  b {{.*}} ; symbol stub for: _objc_msgSend

#--- foo.s
.globl _foo, _objc_msgSend
_foo:
  ret
_objc_msgSend:
  ret

#--- test.s
.section __TEXT,__objc_methname,cstring_literals
lselref1:
  .asciz "foo"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.quad lselref1

.text
.globl _main
.p2align 2
_main:
  bl _foo
  bl _objc_msgSend$foo
  ret
