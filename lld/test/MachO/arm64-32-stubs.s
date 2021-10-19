# REQUIRES: aarch64

## FIXME: This test is very similar to arm64-stubs.s, but has been split into a
## separate file because llvm-objdump doesn't correctly symbolize arm64_32. In
## particular, the "literal pool symbol address" comments are missing (PR49944).

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/test.s -o %t/test.o
# RUN: %lld-watchos -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld-watchos -dylib -install_name @executable_path/libbar.dylib %t/bar.o -o %t/libbar.dylib
# RUN: %lld-watchos -lSystem %t/libfoo.dylib %t/libbar.dylib %t/test.o -o %t/test

# RUN: llvm-objdump --macho -d --no-show-raw-insn --section="__TEXT,__stubs" --section="__TEXT,__stub_helper" %t/test | FileCheck %s

# CHECK:       _main:
# CHECK-NEXT:  bl 0x[[#%x,FOO:]] ; symbol stub for: _foo
# CHECK-NEXT:  bl 0x[[#%x,BAR:]] ; symbol stub for: _bar
# CHECK-NEXT:  ret

# CHECK-LABEL: Contents of (__TEXT,__stubs) section
# CHECK-NEXT:  [[#BAR]]: adrp x16
# CHECK-NEXT:            ldr w16, [x16{{.*}}]
# CHECK-NEXT:            br x16
# CHECK-NEXT:  [[#FOO]]: adrp x16
# CHECK-NEXT:            ldr w16, [x16{{.*}}]
# CHECK-NEXT:            br x16

# CHECK-LABEL: Contents of (__TEXT,__stub_helper) section
# CHECK-NEXT:  [[#%x,HELPER_HEADER:]]: adrp x17
# CHECK-NEXT:                          add x17, x17
# CHECK-NEXT:                          stp x16, x17, [sp, #-16]!
# CHECK-NEXT:                          adrp x16
# CHECK-NEXT:                          ldr w16, [x16]
# CHECK-NEXT:                          br x16
# CHECK-NEXT:                          ldr w16, 0x[[#%x,BAR_BIND_OFF_ADDR:]]
# CHECK-NEXT:                          b 0x[[#HELPER_HEADER]]
# CHECK-NEXT:  [[#BAR_BIND_OFF_ADDR]]: udf #0
# CHECK-NEXT:                          ldr w16, 0x[[#%x,FOO_BIND_OFF_ADDR:]]
# CHECK-NEXT:                          b 0x[[#HELPER_HEADER]]
# CHECK-NEXT:  [[#FOO_BIND_OFF_ADDR]]: udf #11

#--- foo.s
.globl _foo
_foo:

#--- bar.s
.globl _bar
_bar:

#--- test.s
.text
.globl _main

.p2align 2
_main:
  bl _foo
  bl _bar
  ret
