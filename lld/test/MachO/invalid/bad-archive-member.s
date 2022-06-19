# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -dylib -lSystem %t/foo.o -o %t/foo.dylib
# RUN: llvm-ar rcs %t/foo.a %t/foo.dylib
# RUN: not %lld %t/test.o %t/foo.a -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=SYM -DFILE=%t/foo.a
# RUN: not %lld %t/test.o -ObjC %t/foo.a -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=SYM -DFILE=%t/foo.a
# RUN: not %lld %t/test.o -force_load %t/foo.a -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=FORCE-LOAD -DFILE=%t/foo.a
# SYM: error: [[FILE]]: could not get the member defining symbol _foo: foo.dylib has unhandled file type
# FORCE-LOAD: error: [[FILE]]: -force_load failed to load archive member: foo.dylib has unhandled file type

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.globl _main
_main:
  callq _foo
  ret
