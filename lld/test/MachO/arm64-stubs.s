# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -arch arm64 -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -arch arm64 -dylib -install_name @executable_path/libbar.dylib %t/bar.o -o %t/libbar.dylib
# RUN: %lld -arch arm64 -lSystem %t/libfoo.dylib %t/libbar.dylib %t/test.o -o %t/test -no_fixup_chains

# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn --section="__TEXT,__stubs" --section="__TEXT,__stub_helper" %t/test | FileCheck %s

# CHECK:       _main:
# CHECK-NEXT:  bl 0x[[#%x,FOO:]] ; symbol stub for: _foo
# CHECK-NEXT:  bl 0x[[#%x,BAR:]] ; symbol stub for: _bar
# CHECK-NEXT:  ret

# CHECK-LABEL: Contents of (__TEXT,__stubs) section
# CHECK-NEXT:  [[#BAR]]: adrp x16
# CHECK-NEXT:            ldr x16, [x16{{.*}}] ; literal pool symbol address: _bar
# CHECK-NEXT:            br x16
# CHECK-NEXT:  [[#FOO]]: adrp x16
# CHECK-NEXT:            ldr x16, [x16{{.*}}] ; literal pool symbol address: _foo
# CHECK-NEXT:            br x16

# CHECK-LABEL: Contents of (__TEXT,__stub_helper) section
# CHECK-NEXT:  [[#%x,HELPER_HEADER:]]: adrp x17
# CHECK-NEXT:                          add x17, x17
# CHECK-NEXT:                          stp x16, x17, [sp, #-16]!
# CHECK-NEXT:                          adrp x16
# CHECK-NEXT:                          ldr x16, [x16] ; literal pool symbol address: dyld_stub_binder
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
