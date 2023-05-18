# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

## ld64 ignores the __objc_imageinfo section entirely if there is no actual
## ObjC class + category data in the file. LLD doesn't yet do this check, but
## to make this test work for both linkers, I am inserting an appropriate class
## definition into each test file.
# RUN: cat %t/no-category-cls.s   %t/foo-cls.s > %t/no-category-cls-1.s
# RUN: cat %t/with-category-cls.s %t/foo-cls.s > %t/with-category-cls-1.s
# RUN: cat %t/ignored-flags.s     %t/foo-cls.s > %t/ignored-flags-1.s
# RUN: cat %t/invalid-version.s   %t/foo-cls.s > %t/invalid-version-1.s
# RUN: cat %t/invalid-size.s      %t/foo-cls.s > %t/invalid-size-1.s
# RUN: cat %t/swift-version-1.s   %t/foo-cls.s > %t/swift-version-1-1.s
# RUN: cat %t/swift-version-2.s   %t/foo-cls.s > %t/swift-version-2-1.s

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-category-cls-1.s   -o %t/no-category-cls.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/with-category-cls-1.s -o %t/with-category-cls.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ignored-flags-1.s     -o %t/ignored-flags.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/invalid-version-1.s   -o %t/invalid-version.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/swift-version-1-1.s   -o %t/swift-version-1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/swift-version-2-1.s   -o %t/swift-version-2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/invalid-size-1.s      -o %t/invalid-size.o

# RUN: %lld -dylib -lSystem %t/with-category-cls.o -o %t/test-with-cat
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" --syms \
# RUN:   %t/test-with-cat | FileCheck %s --check-prefix=HAS-CAT-CLS \
# RUN:   --implicit-check-not=_discard_me

# RUN: %lld -dylib -lSystem %t/no-category-cls.o -o %t/test-no-cat
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" --syms \
# RUN:   %t/test-no-cat | FileCheck %s --check-prefix=NO-CAT-CLS \
# RUN:   --implicit-check-not=_discard_me

# RUN: %lld -dylib -lSystem %t/no-category-cls.o %t/with-category-cls.o -o %t/test1
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test1 \
# RUN:   | FileCheck %s --check-prefix=NO-CAT-CLS

# RUN: %lld -dylib -lSystem %t/with-category-cls.o %t/ignored-flags.o -o %t/test2
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test2 \
# RUN:   | FileCheck %s --check-prefix=HAS-CAT-CLS

# RUN: %lld -dylib -lSystem %t/no-category-cls.o %t/ignored-flags.o -o %t/test3
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test3 \
# RUN:   | FileCheck %s --check-prefix=NO-CAT-CLS

# RUN: %no-fatal-warnings-lld -dylib -lSystem %t/with-category-cls.o \
# RUN:   %t/invalid-version.o -o %t/test4 2>&1 | FileCheck %s \
# RUN:   --check-prefix=IMAGE-VERSION
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test4 \
# RUN:   | FileCheck %s --check-prefix=NO-CAT-CLS

# RUN: %no-fatal-warnings-lld -dylib -lSystem %t/no-category-cls.o \
# RUN:   %t/invalid-version.o -o %t/test5 2>&1 | FileCheck %s \
# RUN:   --check-prefix=IMAGE-VERSION
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test5 \
# RUN:   | FileCheck %s --check-prefix=NO-CAT-CLS

# RUN: %no-fatal-warnings-lld -dylib -lSystem %t/with-category-cls.o \
# RUN:   %t/invalid-size.o -o %t/test6 2>&1 | FileCheck %s \
# RUN:   --check-prefix=INVALID-SIZE
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" %t/test6 \
# RUN:   | FileCheck %s --check-prefix=NO-CAT-CLS

# RUN: not %lld -dylib -lSystem %t/swift-version-1.o %t/swift-version-2.o -o \
# RUN:   /dev/null 2>&1 | FileCheck %s --check-prefix=SWIFT-MISMATCH-12
# RUN: not %lld -dylib -lSystem %t/swift-version-2.o %t/swift-version-1.o -o \
# RUN:   /dev/null 2>&1 | FileCheck %s --check-prefix=SWIFT-MISMATCH-21

## with-category-cls.o does not have a Swift version (it's set to zero) and
## should be compatible with any Swift version.
# RUN: %lld -dylib -lSystem %t/with-category-cls.o %t/swift-version-1.o -o %t/swift-v1
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" \
# RUN:   %t/swift-v1 | FileCheck %s --check-prefix=SWIFT-V1
# RUN: %lld -dylib -lSystem %t/with-category-cls.o %t/swift-version-2.o -o %t/swift-v2
# RUN: llvm-objdump --macho --section="__DATA_CONST,__objc_imageinfo" \
# RUN:   %t/swift-v2 | FileCheck %s --check-prefix=SWIFT-V2

# HAS-CAT-CLS:       Contents of (__DATA_CONST,__objc_imageinfo) section
# HAS-CAT-CLS:       00 00 00 40 00 00 00
# HAS-CAT-CLS-EMPTY:

# NO-CAT-CLS:       Contents of (__DATA_CONST,__objc_imageinfo) section
# NO-CAT-CLS:       00 00 00 00 00 00 00
# NO-CAT-CLS-EMPTY:

# SWIFT-V1:       Contents of (__DATA_CONST,__objc_imageinfo) section
# SWIFT-V1:       00 00 00 40 01 00 00
# SWIFT-V1-EMPTY:

# SWIFT-V2:       Contents of (__DATA_CONST,__objc_imageinfo) section
# SWIFT-V2:       00 00 00 40 02 00 00
# SWIFT-V2-EMPTY:

# IMAGE-VERSION: warning: {{.*}}invalid-version.o: invalid __objc_imageinfo version

# INVALID-SIZE: warning: {{.*}}invalid-size.o: invalid __objc_imageinfo size

# SWIFT-MISMATCH-12: error: Swift version mismatch: {{.*}}swift-version-1.o has version 1.0 but {{.*}}swift-version-2.o has version 1.1
# SWIFT-MISMATCH-21: error: Swift version mismatch: {{.*}}swift-version-2.o has version 1.1 but {{.*}}swift-version-1.o has version 1.0

#--- no-category-cls.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
## ld64 discards any symbols in this section; we follow suit.
_discard_me:
.long 0
.long 0

#--- with-category-cls.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
_discard_me:
.long 0
.long 0x40 ## "has category class properties" flag

#--- ignored-flags.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
.long 0
## Only the 0x40 flag is carried through to the output binary.
.long (0x40 | 0x20 | 0x4 | 0x2)

#--- invalid-version.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
.long 1 ## only 0 is valid; the flag field below will not be parsed.
.long 0x40

#--- invalid-size.s
.section __DATA,__objc_imageinfo
.long 0

#--- swift-version-1.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
.long 0
.byte 0x40
.byte 0x1 ## Swift version
.short 0

#--- swift-version-2.s
.section __DATA,__objc_imageinfo,regular,no_dead_strip
.long 0
.byte 0x40
.byte 0x2 ## Swift version
.short 0

#--- foo-cls.s
.section __TEXT,__objc_classname,cstring_literals
L_CAT_NAME:
.asciz "barcat"

.section __DATA,__objc_data
.p2align 3
_OBJC_CLASS_$_FooClass:
.space 40

.section __DATA,__objc_const
.p2align 3
__OBJC_$_CATEGORY_FooClass_$_barcat:
.quad L_CAT_NAME
.quad _OBJC_CLASS_$_FooClass
.quad 0
.quad 0
.quad 0
.quad 0
.quad 0
.long 64
.space 4

.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
.quad __OBJC_$_CATEGORY_FooClass_$_barcat
