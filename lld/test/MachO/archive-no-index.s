# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/lib.o %t/lib.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/lib2.o %t/lib2.s
# RUN: llvm-ar crST %t/lib.a %t/lib.o %t/lib2.o
# RUN: %lld %t/main.o %t/lib.a -o %t/out
# RUN: llvm-nm %t/out | FileCheck %s

# CHECK-NOT: T _bar
# CHECK:     T _foo

## Test that every kind of eager load mechanism still works.
# RUN: %lld %t/main.o %t/lib.a -all_load -o %t/all_load
# RUN: llvm-nm %t/all_load | FileCheck %s --check-prefix FORCED-LOAD
# RUN: %lld %t/main.o -force_load %t/lib.a -o %t/force_load
# RUN: llvm-nm %t/force_load | FileCheck %s --check-prefix FORCED-LOAD
# RUN: %lld %t/main.o %t/lib.a -ObjC -o %t/objc
# RUN: llvm-nm %t/objc | FileCheck %s --check-prefix FORCED-LOAD

# FORCED-LOAD: T _bar

#--- lib.s
.global _foo
_foo:
    ret

#--- lib2.s
.section __DATA,__objc_catlist
.quad 0x1234

.section __TEXT,__text
.global _bar
_bar:
    ret

#--- main.s
.global _main
_main:
    call _foo    
    mov $0, %rax
    ret
