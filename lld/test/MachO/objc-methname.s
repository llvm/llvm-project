# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/strings.s -o %t/strings.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -arch arm64 -lSystem -o %t.out %t/strings.o %t/main.o --no-deduplicate-strings

# RUN: llvm-otool -vs __TEXT __cstring %t.out | FileCheck %s --check-prefix=CSTRING
# RUN: llvm-otool -vs __TEXT __objc_methname %t.out | FileCheck %s --check-prefix=METHNAME

# RUN: %lld -arch arm64 -lSystem -o %t/duplicates %t/strings.o %t/strings.o %t/main.o

# RUN: llvm-otool -vs __TEXT __cstring %t/duplicates | FileCheck %s --check-prefix=CSTRING
# RUN: llvm-otool -vs __TEXT __objc_methname %t/duplicates | FileCheck %s --check-prefix=METHNAME

# CSTRING: Contents of (__TEXT,__cstring) section
# CSTRING-NEXT: existing-cstring
# CSTIRNG-EMPTY:

# METHNAME: Contents of (__TEXT,__objc_methname) section
# METHNAME-NEXT: existing_methname
# METHNAME-NEXT: synthetic_methname
# METHNAME-EMPTY:

#--- strings.s
.cstring
.p2align 2
  .asciz "existing-cstring"

.section __TEXT,__objc_methname,cstring_literals
  .asciz  "existing_methname"

#--- main.s
.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl  _objc_msgSend$existing_methname
  bl  _objc_msgSend$synthetic_methname
  ret
