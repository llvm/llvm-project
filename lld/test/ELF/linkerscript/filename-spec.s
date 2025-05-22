# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: cd %t && mkdir dir0 dir1 dir2
# RUN: llvm-mc -filetype=obj -triple=x86_64 tx.s -o tx.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ty.s -o ty.o

# RUN: echo 'SECTIONS{.foo :{ KEEP(:*x.o(.foo)) KEEP(*y.o(.foo)) }}' > 1.t
# RUN: ld.lld -o 1 -T 1.t tx.o ty.o
# RUN: llvm-objdump -s 1 | FileCheck --check-prefix=FIRSTY %s
# FIRSTY:      Contents of section .foo:
# FIRSTY-NEXT:   01000000 00000000 11000000 00000000

# RUN: echo 'SECTIONS{.foo :{ KEEP(*y.o(.foo)) KEEP(*x.o(.foo)) }}' > 2.t
# RUN: ld.lld -o 2 -T 2.t tx.o ty.o
# RUN: llvm-objdump -s 2 | FileCheck --check-prefix=SECONDFIRST %s
# SECONDFIRST:      Contents of section .foo:
# SECONDFIRST-NEXT:   11000000 00000000 01000000 00000000

## Now the same tests but without KEEP. Checking that file name inside
## KEEP is parsed fine.
# RUN: echo 'SECTIONS{.foo :{ :*x.o(.foo) *y.o(.foo) }}' > 3.t
# RUN: ld.lld -o 3 -T 3.t tx.o ty.o
# RUN: llvm-objdump -s 3 | FileCheck --check-prefix=FIRSTY %s

# RUN: echo 'SECTIONS{.foo :{ *y.o(.foo) *x.o(.foo) }}' > 4.t
# RUN: ld.lld -o 4 -T 4.t tx.o ty.o
# RUN: llvm-objdump -s 4 | FileCheck --check-prefix=SECONDFIRST %s

# RUN: cp tx.o dir0/filename-spec1.o
# RUN: cp ty.o dir0/filename-spec2.o

# RUN: echo 'SECTIONS{.foo :{ "dir0/filename-spec2.o"(.foo) "dir0/filename-spec1.o"(.foo) }}' > 5.t
# RUN: ld.lld -o 5 -T 5.t dir0/filename-spec1.o dir0/filename-spec2.o
# RUN: llvm-objdump -s 5 | FileCheck --check-prefix=SECONDFIRST %s

# RUN: echo 'SECTIONS{.foo :{ "dir0/filename-spec1.o"(.foo) "dir0/filename-spec2.o"(.foo) }}' > 6.t
# RUN: ld.lld -o 6 -T 6.t dir0/filename-spec1.o dir0/filename-spec2.o
# RUN: llvm-objdump -s 6 | FileCheck --check-prefix=FIRSTY %s

# RUN: cp tx.o dir1/filename-spec1.o
# RUN: cp ty.o dir2/filename-spec2.o
# RUN: llvm-ar rc dir1/lib1.a dir1/filename-spec1.o
# RUN: llvm-ar rc dir2/lib2.a dir2/filename-spec2.o
# RUN: llvm-ar rc combined.a tx.o ty.o

## Verify matching of archive library names.
# RUN: echo 'SECTIONS{.foo :{ *lib2*(.foo) *lib1*(.foo) }}' > 7.t
# RUN: ld.lld -o 7 -T 7.t --whole-archive \
# RUN:   dir1/lib1.a dir2/lib2.a
# RUN: llvm-objdump -s 7 | FileCheck --check-prefix=SECONDFIRST %s

## Verify matching directories.
# RUN: echo 'SECTIONS{.foo :{  *dir2*(.foo) *dir1*(.foo) }}' > 8.t
# RUN: ld.lld -o 8 -T 8.t --whole-archive \
# RUN:   dir1/lib1.a dir2/lib2.a
# RUN: llvm-objdump -s 8 | FileCheck --check-prefix=SECONDFIRST %s

## Verify matching of archive library names in KEEP.
# RUN: echo 'SECTIONS{.foo :{ KEEP(*lib2.a:(.foo)) KEEP(*lib1*(.foo)) }}' > 9.t
# RUN: ld.lld -o 9 -T 9.t --whole-archive \
# RUN:   dir1/lib1.a dir2/lib2.a
# RUN: llvm-objdump -s 9 | FileCheck --check-prefix=SECONDFIRST %s

## Verify matching directories in KEEP.
# RUN: echo 'SECTIONS{.foo :{ KEEP(*dir2*(.foo)) KEEP(*dir1*(.foo)) }}' > 10.t
# RUN: ld.lld -o 10 -T 10.t --whole-archive \
# RUN:   dir1/lib1.a dir2/lib2.a
# RUN: llvm-objdump -s 10 | FileCheck --check-prefix=SECONDFIRST %s

## () can appear in a quoted filename pattern.
# RUN: cp dir1/lib1.a 'lib1().a'
# RUN: echo 'SECTIONS{.foo :{ KEEP(*dir2*(.foo)) KEEP("lib1().a"(.foo)) }}' > 11.t
# RUN: ld.lld -o 11 -T 11.t --whole-archive 'lib1().a' dir2/lib2.a
# RUN: llvm-objdump -s 11 | FileCheck --check-prefix=SECONDFIRST %s

## Verify that matching files excluded from an archive will not match files within one.
# RUN: echo 'SECTIONS{.foo :{ KEEP(:*x.o(.foo)) KEEP(*y.o(.foo)) KEEP(*x.o(.foo)) }}' > 12.t
# RUN: ld.lld -o 12 -T 12.t --whole-archive combined.a
# RUN: llvm-objdump -s 12 | FileCheck --check-prefix=SECONDFIRST %s

#--- tx.s
.global _start
_start:
 nop

.section .foo,"a"
 .quad 1

#--- ty.s
.section .foo,"a"
  .quad 0x11
